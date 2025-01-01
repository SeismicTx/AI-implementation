import pandas as pd
import numpy as np
from pathlib import Path
import re


class AntibodyDataProcessor:
    def __init__(self, data_dir):
        """
        Initialize the data processor with the directory containing CSV files.

        Args:
            data_dir (str): Path to directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}

    def load_data(self):
        """Load all CSV files from the data directory."""
        for file_path in self.data_dir.glob('*.csv'):
            name = file_path.stem
            self.datasets[name] = pd.read_csv(file_path)

    def _standardize_column_names(self, df, prefix=''):
        """
        Standardize column names by removing special characters and units,
        and adding prefixes to avoid column name conflicts.
        """

        def clean_name(col):
            if col != 'antibody_id':
                # Remove units in parentheses or after _
                col = re.sub(r'\([^)]*\)', '', col)
                col = re.sub(r'_[^_]*/', '_', col)
                # Remove special characters and standardize separators
                col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                # Remove duplicate underscores
                col = re.sub(r'_+', '_', col)
                # Remove trailing underscores
                col = col.strip('_').lower()
                return f"{prefix}{col}" if prefix else col
            else:
                return col

        return df.rename(columns=lambda x: clean_name(x))

    def _convert_to_numeric(self, df):
        """Convert string columns to numeric where possible."""
        for col in df.columns:
            # Skip clearly non-numeric columns
            if col in ['notes', 'comments', 'analyst', 'process', 'method', 'appearance', 'storage',
                       'format', 'species', 'isotype', 'cell_line', 'batch', 'buffer', 'test_method']:
                continue

            if df[col].dtype == 'object':
                # Try converting to numeric, handling percentage signs
                try:
                    df[col] = df[col].str.replace('%', '').astype(float)
                except:
                    pass
        return df

    def _handle_missing_values(self, df):
        """Handle missing values based on column type."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # For numeric columns, fill with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # For categorical columns, fill with 'unknown'
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        return df

    def _standardize_antibody_ids(self):
        """Standardize antibody IDs across all datasets."""
        id_columns = {
            'asec': 'ID',
            'binding_affinity': 'Sample_ID',
            'bioactivity': 'Sample_ID',
            'charge_variants': 'SampleName',
            'endotoxin': 'ID',
            'expression_yields': 'Antibody_ID',
            'glycan_profiling': 'Ab_ID',
            'sequences': 'Antibody_ID',
            'stability_timecourse': 'AntibodyName',
            'thermostability': 'AntibodyID'
        }

        # Standardize ID column names
        for dataset_name, id_col in id_columns.items():
            if dataset_name in self.datasets:
                self.datasets[dataset_name] = self.datasets[dataset_name].rename(
                    columns={id_col: 'antibody_id'})

    def process_datasets(self):
        """Process all datasets with standardization and cleaning steps."""
        # First standardize IDs
        self._standardize_antibody_ids()

        # Process each dataset
        for name, df in self.datasets.items():
            # Add prefix to avoid column name conflicts
            df = self._standardize_column_names(df, prefix=f"{name}_")
            # Convert string numbers to numeric
            df = self._convert_to_numeric(df)
            # Handle missing values
            df = self._handle_missing_values(df)
            # Update the processed dataset
            self.datasets[name] = df

    def merge_datasets(self):
        """
        Merge all datasets on antibody_id.
        Returns:
            pd.DataFrame: Merged dataset
        """
        # Start with sequences as base if available, otherwise use first available dataset
        base_dataset = 'sequences'
        if base_dataset not in self.datasets:
            base_dataset = list(self.datasets.keys())[0]

        merged_df = self.datasets[base_dataset]

        # Merge rest of datasets
        for name, df in self.datasets.items():
            if name != base_dataset:
                merged_df = merged_df.merge(df, on='antibody_id', how='outer')

        return merged_df

    def prepare_for_ml(self, merged_df):
        """
        Prepare the merged dataset for machine learning.

        Args:
            merged_df (pd.DataFrame): Merged dataset

        Returns:
            pd.DataFrame: ML-ready dataset
        """
        # Drop columns that aren't useful for ML
        cols_to_drop = [col for col in merged_df.columns
                        if any(x in col.lower() for x in ['notes', 'comments', 'analyst'])]
        ml_df = merged_df.drop(columns=cols_to_drop)

        # Convert categorical variables to dummy variables
        categorical_columns = ml_df.select_dtypes(include=['object']).columns
        ml_df = pd.get_dummies(ml_df, columns=categorical_columns)

        # Final missing value handling
        ml_df = ml_df.fillna(ml_df.median())

        return ml_df


def main():
    # Initialize processor
    processor = AntibodyDataProcessor('../data')

    # Load and process data
    processor.load_data()
    processor.process_datasets()

    # Merge datasets
    merged_df = processor.merge_datasets()

    # Prepare for ML
    ml_ready_df = processor.prepare_for_ml(merged_df)

    # Save processed datasets
    merged_df.to_csv('merged_antibody_data.csv', index=False)
    ml_ready_df.to_csv('ml_ready_antibody_data.csv', index=False)

    print(f"Final dataset shape: {ml_ready_df.shape}")
    print("\nFeature names:")
    print(ml_ready_df.columns.tolist())

if __name__ == '__main__':
    main()