#!/usr/bin/env python3

from benchling_sdk.auth import RequestsWithBenchlingAuth
from benchling_sdk.benchling import Benchling
from benchling_sdk.helpers import (
    filter_to_dict,
    get_entity_schema_id_from_registry_id
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os
import json
from dataclasses import dataclass
from retry import retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchlingConfig:
    """Configuration for Benchling API access"""
    api_key: str
    tenant_name: str
    registry_id: str
    schema_name: str = "Antibody_Properties"
    days_lookback: int = 7


class BenchlingDataFetcher:
    def __init__(self, config: BenchlingConfig):
        """Initialize Benchling API client"""
        self.config = config
        self.benchling = Benchling(
            url=f"https://{config.tenant_name}.benchling.com/api/v2/",
            api_key=config.api_key
        )
        self._init_schemas()

    def _init_schemas(self):
        """Initialize schema IDs for entities"""
        try:
            schemas = self.benchling.entity_schemas.list()
            self.antibody_schema_id = next(
                (s.id for s in schemas if s.name == self.config.schema_name),
                None
            )
            if not self.antibody_schema_id:
                raise ValueError(f"Schema {self.config.schema_name} not found")
        except Exception as e:
            logger.error(f"Failed to initialize schemas: {str(e)}")
            raise

    @retry(tries=3, delay=2, backoff=2)
    def fetch_antibody_data(self, days_lookback: Optional[int] = None) -> List[Dict]:
        """
        Fetch antibody data from Benchling

        Args:
            days_lookback: Optional override for config days_lookback

        Returns:
            List of antibody records with sequence and property data
        """
        lookback = days_lookback or self.config.days_lookback
        start_date = datetime.now() - timedelta(days=lookback)

        try:
            # Get antibody entities
            entities = self.benchling.entities.list(
                schema_id=self.antibody_schema_id,
                modified_at={"$gte": start_date.isoformat()}
            )

            antibody_data = []
            for entity in entities:
                # Get associated sequence data
                sequence = self.benchling.sequences.get(entity.entityRegistryId)

                # Extract relevant fields
                record = {
                    'sequence_id': entity.entityRegistryId,
                    'name': entity.name,
                    'heavy_chain': sequence.heavyChainSequence,
                    'light_chain': sequence.lightChainSequence,
                    'binding_affinity': entity.fields.get('binding_affinity'),
                    'expression_level': entity.fields.get('expression_level'),
                    'thermal_stability': entity.fields.get('thermal_stability'),
                    'created_at': entity.created_at,
                    'modified_at': entity.modified_at
                }

                # Validate required fields
                if all(record[field] for field in ['heavy_chain', 'light_chain', 'binding_affinity']):
                    antibody_data.append(record)
                else:
                    logger.warning(f"Skipping incomplete record: {entity.entityRegistryId}")

            return antibody_data

        except Exception as e:
            logger.error(f"Error fetching antibody data: {str(e)}")
            raise

    def save_to_csv(self, data: List[Dict], output_path: str):
        """Save antibody data to CSV"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} records to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


class BenchlingDataUploader:
    def __init__(self, config: BenchlingConfig):
        """Initialize Benchling API client for uploading results"""
        self.config = config
        self.benchling = Benchling(
            url=f"https://{config.tenant_name}.benchling.com/api/v2/",
            api_key=config.api_key
        )

    @retry(tries=3, delay=2, backoff=2)
    def upload_predictions(self, predictions: List[Dict]):
        """
        Upload model predictions back to Benchling

        Args:
            predictions: List of dictionaries containing:
                - sequence_id: Benchling entity registry ID
                - predicted_affinity: Model prediction
                - confidence_score: Model confidence
        """
        try:
            for pred in predictions:
                # Update entity with predictions
                entity = self.benchling.entities.get(pred['sequence_id'])

                updates = {
                    'fields': {
                        'predicted_affinity': pred['predicted_affinity'],
                        'prediction_confidence': pred['confidence_score'],
                        'prediction_date': datetime.now().isoformat()
                    }
                }

                self.benchling.entities.update(
                    entity.id,
                    updates
                )

                logger.info(f"Updated predictions for {pred['sequence_id']}")

        except Exception as e:
            logger.error(f"Error uploading predictions: {str(e)}")
            raise


def main():
    # Example usage
    config = BenchlingConfig(
        api_key=os.environ['BENCHLING_API_KEY'],
        tenant_name=os.environ['BENCHLING_TENANT'],
        registry_id=os.environ['BENCHLING_REGISTRY_ID']
    )

    # Fetch data
    fetcher = BenchlingDataFetcher(config)
    data = fetcher.fetch_antibody_data()
    fetcher.save_to_csv(data, 'antibody_data.csv')

    # Example of uploading predictions
    uploader = BenchlingDataUploader(config)
    predictions = [
        {
            'sequence_id': 'seq_123',
            'predicted_affinity': 0.85,
            'confidence_score': 0.92
        }
    ]
    uploader.upload_predictions(predictions)


if __name__ == '__main__':
    main()

# Tests for Benchling integration
import pytest
from unittest.mock import Mock, patch


def test_benchling_data_fetcher():
    config = BenchlingConfig(
        api_key='test_key',
        tenant_name='test_tenant',
        registry_id='test_registry'
    )

    # Mock Benchling API responses
    mock_entity = Mock(
        entityRegistryId='test_seq',
        name='Test Antibody',
        fields={'binding_affinity': 0.75}
    )
    mock_sequence = Mock(
        heavyChainSequence='QVQLVQ',
        lightChainSequence='DIQMTQ'
    )

    with patch('benchling_sdk.benchling.Benchling') as MockBenchling:
        mock_client = Mock()
        mock_client.entities.list.return_value = [mock_entity]
        mock_client.sequences.get.return_value = mock_sequence
        MockBenchling.return_value = mock_client

        fetcher = BenchlingDataFetcher(config)
        data = fetcher.fetch_antibody_data()

        assert len(data) == 1
        assert data[0]['sequence_id'] == 'test_seq'
        assert data[0]['heavy_chain'] == 'QVQLVQ'