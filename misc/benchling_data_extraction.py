import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import yaml
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import re
from tenacity import retry, stop_after_attempt, wait_exponential
import boto3
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlalchemy as sa


class BenchlingDataExtractor:
    def __init__(self, config_path: str):
        """
        Initialize the Benchling data extractor with configuration.

        Args:
            config_path: Path to YAML configuration file containing API credentials
                        and schema mappings
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_api_client()
        self.setup_aws_clients()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('benchling_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_api_client(self):
        """Setup Benchling API client with authentication."""
        self.api_base_url = f"https://{self.config['benchling']['tenant']}.benchling.com/api/v2"
        self.headers = {
            "Authorization": f"Bearer {self.config['benchling']['api_token']}",
            "Content-Type": "application/json"
        }

    def setup_aws_clients(self):
        """Initialize AWS service clients."""
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.config['aws']['access_key'],
            aws_secret_access_key=self.config['aws']['secret_key'],
            region_name=self.config['aws']['region']
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_api_request(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None) -> Dict:
        """Make API request to Benchling with retry logic."""
        url = f"{self.api_base_url}/{endpoint}"

        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            params=params,
            json=data
        )

        response.raise_for_status()
        return response.json()

    def get_entry_schemas(self) -> List[Dict]:
        """Retrieve all entry schemas from Benchling."""
        try:
            response = self._make_api_request('entry-schemas')
            return response['entrySchemas']
        except Exception as e:
            self.logger.error(f"Error retrieving entry schemas: {str(e)}")
            raise

    def get_entries(self, schema_id: str, start_date: datetime = None) -> List[Dict]:
        """
        Retrieve entries for a specific schema with optional date filtering.

        Args:
            schema_id: Benchling schema ID
            start_date: Optional start date for filtering entries
        """
        entries = []
        next_token = None

        try:
            while True:
                params = {
                    'schemaId': schema_id,
                    'pageSize': 50
                }

                if next_token:
                    params['nextToken'] = next_token

                if start_date:
                    params['modifiedAt'] = {
                        '$gte': start_date.isoformat()
                    }

                response = self._make_api_request('entries', params=params)
                entries.extend(response['entries'])

                next_token = response.get('nextToken')
                if not next_token:
                    break

            return entries

        except Exception as e:
            self.logger.error(f"Error retrieving entries for schema {schema_id}: {str(e)}")
            raise

    def extract_results_table(self, entry_id: str) -> List[Dict]:
        """Extract Results table data from an entry."""
        try:
            response = self._make_api_request(f'entries/{entry_id}/results-tables')
            return response['resultsTables']
        except Exception as e:
            self.logger.error(f"Error extracting Results table from entry {entry_id}: {str(e)}")
            raise

    def parse_eln_content(self, entry: Dict) -> Dict:
        """
        Parse structured and unstructured data from ELN entry content.

        Args:
            entry: Benchling entry dictionary
        """
        parsed_data = {}

        try:
            # Parse structured fields
            for field in entry.get('fields', []):
                field_name = field['name']
                field_value = field.get('value')
                parsed_data[field_name] = field_value

            # Parse text content for embedded tables and structured data
            if 'text' in entry:
                soup = BeautifulSoup(entry['text'], 'html.parser')

                # Extract tables
                tables = soup.find_all('table')
                for i, table in enumerate(tables):
                    table_data = []
                    headers = []

                    # Get headers
                    header_row = table.find('tr')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]

                    # Get data rows
                    for row in table.find_all('tr')[1:]:
                        cells = [td.text.strip() for td in row.find_all('td')]
                        if len(cells) == len(headers):
                            table_data.append(dict(zip(headers, cells)))

                    if table_data:
                        parsed_data[f'embedded_table_{i}'] = table_data

                # Extract structured patterns (e.g., key-value pairs)
                text_content = soup.get_text()

                # Example pattern matching for key-value pairs
                kv_patterns = [
                    r'(?P<key>[A-Za-z\s]+):\s*(?P<value>[\d.]+(?:\s*[A-Za-z]+)?)',
                    r'(?P<key>[A-Za-z\s]+)=\s*(?P<value>[\d.]+(?:\s*[A-Za-z]+)?)'
                ]

                for pattern in kv_patterns:
                    matches = re.finditer(pattern, text_content)
                    for match in matches:
                        key = match.group('key').strip()
                        value = match.group('value').strip()
                        parsed_data[f'extracted_{key}'] = value

            return parsed_data

        except Exception as e:
            self.logger.error(f"Error parsing ELN content: {str(e)}")
            raise

    def _standardize_data(self, data: Dict, schema_mapping: Dict) -> Dict:
        """
        Standardize extracted data based on schema mapping.

        Args:
            data: Raw extracted data
            schema_mapping: Schema mapping configuration
        """
        standardized = {}

        try:
            # Apply column mappings
            for raw_key, std_key in schema_mapping.get('column_mapping', {}).items():
                if raw_key in data:
                    standardized[std_key] = data[raw_key]

            # Apply data type conversions
            for field, dtype in schema_mapping.get('data_types', {}).items():
                if field in standardized:
                    try:
                        if dtype == 'float':
                            standardized[field] = pd.to_numeric(standardized[field], errors='coerce')
                        elif dtype == 'date':
                            standardized[field] = pd.to_datetime(standardized[field], errors='coerce')
                        elif dtype == 'string':
                            standardized[field] = str(standardized[field])
                    except Exception as e:
                        self.logger.warning(f"Error converting {field} to {dtype}: {str(e)}")

            # Apply validation rules
            for field, rules in schema_mapping.get('validation_rules', {}).items():
                if field in standardized:
                    if 'range' in rules:
                        min_val, max_val = rules['range']
                        if isinstance(standardized[field], (int, float)):
                            standardized[field] = np.clip(standardized[field], min_val, max_val)
                    if 'categories' in rules:
                        if standardized[field] not in rules['categories']:
                            standardized[field] = None

            return standardized

        except Exception as e:
            self.logger.error(f"Error standardizing data: {str(e)}")
            raise

    def process_entries(self, schema_id: str, start_date: Optional[datetime] = None) -> List[Dict]:
        """
        Process all entries for a schema, extracting and standardizing data.

        Args:
            schema_id: Benchling schema ID
            start_date: Optional start date for filtering entries
        """
        processed_data = []

        try:
            # Get entries
            entries = self.get_entries(schema_id, start_date)

            # Get schema mapping
            schema_mapping = self.config['schema_mapping'].get(schema_id, {})

            for entry in entries:
                entry_data = {}

                # Extract Results tables
                results_tables = self.extract_results_table(entry['id'])
                if results_tables:
                    for table in results_tables:
                        table_data = pd.DataFrame(table['data'])
                        entry_data.update(table_data.to_dict('records')[0])

                # Parse ELN content
                eln_data = self.parse_eln_content(entry)
                entry_data.update(eln_data)

                # Standardize data
                standardized_data = self._standardize_data(entry_data, schema_mapping)

                # Add metadata
                standardized_data.update({
                    'entry_id': entry['id'],
                    'schema_id': schema_id,
                    'created_at': entry['createdAt'],
                    'modified_at': entry['modifiedAt']
                })

                processed_data.append(standardized_data)

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing entries for schema {schema_id}: {str(e)}")
            raise

    def save_to_s3(self, data: List[Dict], schema_id: str):
        """Save processed data to S3."""
        try:
            df = pd.DataFrame(data)

            # Convert to parquet
            parquet_buffer = df.to_parquet()

            # Generate S3 key with partitioning
            timestamp = datetime.now().strftime('%Y/%m/%d/%H')
            s3_key = f"benchling_data/{schema_id}/{timestamp}/data.parquet"

            # Upload to S3
            self.s3.put_object(
                Bucket=self.config['aws']['data_lake_bucket'],
                Key=s3_key,
                Body=parquet_buffer
            )

            self.logger.info(f"Saved {len(data)} records to s3://{self.config['aws']['data_lake_bucket']}/{s3_key}")

        except Exception as e:
            self.logger.error(f"Error saving data to S3: {str(e)}")
            raise

    def load_to_database(self, data: List[Dict], schema_id: str):
        """Load processed data into database."""
        try:
            df = pd.DataFrame(data)

            # Create database connection
            engine = sa.create_engine(self.config['database']['connection_string'])

            # Load data
            table_name = f"benchling_{schema_id.lower()}"
            df.to_sql(
                table_name,
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )

            self.logger.info(f"Loaded {len(data)} records into {table_name}")

        except Exception as e:
            self.logger.error(f"Error loading data to database: {str(e)}")
            raise


def main():
    # Initialize extractor
    extractor = BenchlingDataExtractor('config.yaml')

    # Get all schemas
    schemas = extractor.get_entry_schemas()

    # Process each schema
    for schema in schemas:
        schema_id = schema['id']

        # Get last 24 hours of data
        start_date = datetime.now() - timedelta(days=1)

        # Process entries
        processed_data = extractor.process_entries(schema_id, start_date)

        if processed_data:
            # Save to S3
            extractor.save_to_s3(processed_data, schema_id)

            # Load to database
            extractor.load_to_database(processed_data, schema_id)


if __name__ == "__main__":
    main()