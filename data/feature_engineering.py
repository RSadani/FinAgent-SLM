import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Optional

class FeatureEngineer:
    """
    Feature engineering pipeline for financial transaction and tabular data.
    Provides methods to add statistical, velocity, graph-based, and time-based features.
    """

    @staticmethod
    def add_statistical_features(df: pd.DataFrame, amount_col: str = 'amount') -> pd.DataFrame:
        """Adds statistical features like z-scores, round numbers, and rolling stats."""
        logger.info(f"Adding statistical features based on {amount_col}...")
        df = df.copy()
        
        if amount_col in df.columns:
            # Z-score of amount
            mean_amt = df[amount_col].mean()
            std_amt = df[amount_col].std()
            df[f'{amount_col}_zscore'] = (df[amount_col] - mean_amt) / (std_amt + 1e-8)
            
            # Is round number (common in fraud)
            df['is_round_amount'] = (df[amount_col] % 100 == 0).astype(int)
            
            # Log transform for heavily skewed financial amounts
            df[f'{amount_col}_log'] = np.log1p(df[amount_col])
            
        return df

    @staticmethod
    def add_velocity_features(df: pd.DataFrame, entity_col: str, time_col: str, amount_col: str) -> pd.DataFrame:
        """Adds transaction velocity features (e.g., time since last transaction)."""
        logger.info(f"Adding velocity features for {entity_col}...")
        df = df.copy()
        
        if all(c in df.columns for c in [entity_col, time_col, amount_col]):
            df = df.sort_values(by=[entity_col, time_col])
            
            # Time since last transaction for the entity
            df['time_since_last_tx'] = df.groupby(entity_col)[time_col].diff().fillna(-1)
            
            # Rolling transaction count and sum in recent history (simplification using expanding/shift)
            df['tx_count_history'] = df.groupby(entity_col).cumcount()
            
        return df

    @staticmethod
    def add_graph_features(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        """Adds approximation of graph-based features (degree centrality proxy)."""
        logger.info("Adding graph-based features (degree proxy)...")
        df = df.copy()
        
        if source_col in df.columns and target_col in df.columns:
            # Out-degree proxy: How many distinct targets this source has sent to
            out_degree = df.groupby(source_col)[target_col].transform('nunique')
            df['source_out_degree'] = out_degree
            
            # In-degree proxy: How many distinct sources this target has received from
            in_degree = df.groupby(target_col)[source_col].transform('nunique')
            df['target_in_degree'] = in_degree
            
        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Adds time-based behavioral features."""
        logger.info(f"Adding time-based features from {time_col}...")
        df = df.copy()
        
        if time_col in df.columns:
            # Assuming time_col is 'step' representing hours (as in PaySim)
            df['hour_of_day'] = df[time_col] % 24
            df['day_of_week'] = (df[time_col] // 24) % 7
            
            # Flag for unusual hours (e.g., late night 1 AM to 5 AM)
            df['is_unusual_hour'] = df['hour_of_day'].apply(lambda x: 1 if 1 <= x <= 5 else 0)
            
        return df

    @staticmethod
    def add_behavioral_deviation_features(df: pd.DataFrame, entity_col: str, amount_col: str) -> pd.DataFrame:
        """Calculates deviation from an entity's typical behavior."""
        logger.info("Adding behavioral deviation features...")
        df = df.copy()
        
        if entity_col in df.columns and amount_col in df.columns:
            entity_means = df.groupby(entity_col)[amount_col].transform('mean')
            entity_stds = df.groupby(entity_col)[amount_col].transform('std').fillna(0)
            
            # Deviation from user's own mean
            df['tx_amount_user_deviation'] = (df[amount_col] - entity_means) / (entity_stds + 1e-8)
            
        return df
