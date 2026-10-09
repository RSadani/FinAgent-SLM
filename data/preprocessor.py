import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imblearn not installed. SMOTE will not be available.")

class DataPreprocessor:
    """
    Data preprocessing pipeline for financial datasets.
    Handles missing values, encoding, scaling, oversampling, and feature selection.
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard', 
                 apply_smote: bool = False,
                 feature_selection_k: Optional[int] = None):
        self.scaling_method = scaling_method
        self.apply_smote = apply_smote
        self.feature_selection_k = feature_selection_k
        
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.encoders = {}
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=feature_selection_k) if feature_selection_k else None
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.selected_features = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """Fit the preprocessor on the dataset."""
        logger.info("Fitting DataPreprocessor...")
        
        # Identify column types
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit imputers
        if self.numeric_cols:
            self.num_imputer.fit(X[self.numeric_cols])
        if self.categorical_cols:
            self.cat_imputer.fit(X[self.categorical_cols])
            
        # Fit encoders
        for col in self.categorical_cols:
            self.encoders[col] = LabelEncoder()
            # Fill NaNs temporarily to fit encoder
            temp_col = X[col].fillna('Missing_Unknown')
            self.encoders[col].fit(temp_col.astype(str))
            
        # Fit scaler
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
            
        # Fit feature selector if requested
        if self.feature_selection_k and y is not None:
            logger.info(f"Selecting top {self.feature_selection_k} features using mutual information...")
            X_temp = self._transform_without_selection(X)
            self.feature_selector.fit(X_temp, y)
            mask = self.feature_selector.get_support()
            self.selected_features = X_temp.columns[mask].tolist()
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset using fitted transformers."""
        logger.info("Transforming data...")
        X_out = self._transform_without_selection(X)
        
        if self.feature_selection_k and self.selected_features:
            X_out = X_out[self.selected_features]
            
        return X_out

    def _transform_without_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method without feature selection step."""
        X_out = X.copy()
        
        # Impute
        if self.numeric_cols:
            X_out[self.numeric_cols] = self.num_imputer.transform(X_out[self.numeric_cols])
        if self.categorical_cols:
            X_out[self.categorical_cols] = self.cat_imputer.transform(X_out[self.categorical_cols])
            
        # Encode
        for col in self.categorical_cols:
            # Handle unseen labels gracefully by mapping them to an 'Unknown' or using a fallback
            X_out[col] = X_out[col].astype(str)
            # Find classes that were not in the fitted encoder
            unseen = ~X_out[col].isin(self.encoders[col].classes_)
            if unseen.any():
                logger.warning(f"Unseen labels found in column {col}, masking them.")
                # Fallback: assign to the most frequent or first class
                X_out.loc[unseen, col] = self.encoders[col].classes_[0] 
            
            X_out[col] = self.encoders[col].transform(X_out[col])
            
        # Scale numeric features
        if self.numeric_cols:
            X_out[self.numeric_cols] = self.scaler.transform(X_out[self.numeric_cols])
            
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit and transform the dataset, optionally applying SMOTE."""
        self.fit(X, y)
        X_out = self.transform(X)
        
        if self.apply_smote and HAS_IMBLEARN:
            logger.info("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_out, y)
            return X_resampled, y_resampled
            
        return X_out, y
