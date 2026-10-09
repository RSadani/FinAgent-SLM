import os
import torch
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split

from .feature_engineering import FeatureEngineer
from .preprocessor import DataPreprocessor

class BaseFinancialDataset(Dataset):
    """Base class for financial datasets in PyTorch."""
    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PaySimDataset(BaseFinancialDataset):
    """
    PaySim Synthetic Mobile Money Dataset for Fraud Detection.
    Generates synthetic data mirroring the Kaggle dataset if CSV is unavailable.
    """
    @classmethod
    def load_or_generate(cls, 
                         csv_path: Optional[str] = None, 
                         n_samples: int = 100000) -> Tuple[pd.DataFrame, pd.Series]:
        if csv_path and os.path.exists(csv_path):
            logger.info(f"Loading PaySim from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            logger.info(f"Generating {n_samples} synthetic PaySim records...")
            df = cls.generate_synthetic(n_samples)
            
        # Feature Engineering
        df = FeatureEngineer.add_statistical_features(df, 'amount')
        df = FeatureEngineer.add_time_features(df, 'step')
        
        # Balance diff features
        df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
        
        labels = df['isFraud']
        features = df.drop(columns=['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'])
        
        return features, labels

    @classmethod
    def generate_synthetic(cls, n_samples: int = 100000) -> pd.DataFrame:
        np.random.seed(42)
        
        # Standard variables
        step = np.random.randint(1, 744, n_samples)
        types = np.random.choice(['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'], 
                                 n_samples, p=[0.22, 0.35, 0.10, 0.32, 0.01])
        amount = np.random.lognormal(mean=5, sigma=2, size=n_samples)
        
        # Balances
        oldbalanceOrg = np.random.lognormal(mean=6, sigma=2.5, size=n_samples)
        newbalanceOrig = np.where(np.isin(types, ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']),
                                  np.maximum(0, oldbalanceOrg - amount),
                                  oldbalanceOrg + amount)
                                  
        oldbalanceDest = np.random.lognormal(mean=7, sigma=2, size=n_samples)
        newbalanceDest = np.where(np.isin(types, ['CASH_IN', 'TRANSFER', 'CASH_OUT']),
                                  oldbalanceDest + amount,
                                  oldbalanceDest)

        # Base dataframe
        df = pd.DataFrame({
            'step': step,
            'type': types,
            'amount': amount,
            'nameOrig': [f'C{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'nameDest': [f'M{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'isFraud': 0,
            'isFlaggedFraud': 0
        })

        # Inject realistic fraud patterns (approx 1.3%)
        # Pattern 1: Massive transfers draining accounts
        fraud_idx_1 = df[(df['type'] == 'TRANSFER') & (df['amount'] > 500000)].sample(frac=0.3, random_state=1).index
        # Pattern 2: Rapid successive cashouts
        fraud_idx_2 = df[(df['type'] == 'CASH_OUT') & (df['amount'] == df['oldbalanceOrg']) & (df['amount'] > 1000)].sample(frac=0.5, random_state=2).index
        # Pattern 3: Round amounts in transfers
        fraud_idx_3 = df[(df['type'] == 'TRANSFER') & (df['amount'] % 1000 == 0) & (df['amount'] > 50000)].sample(frac=0.2, random_state=3).index

        fraud_indices = list(set(fraud_idx_1) | set(fraud_idx_2) | set(fraud_idx_3))
        # Keep fraud rate to roughly 1.3%
        target_fraud_count = int(n_samples * 0.013)
        if len(fraud_indices) > target_fraud_count:
            fraud_indices = np.random.choice(fraud_indices, target_fraud_count, replace=False)
            
        df.loc[fraud_indices, 'isFraud'] = 1
        
        # Rule for flagged fraud (amount > 200k in transfer)
        df.loc[(df['isFraud'] == 1) & (df['type'] == 'TRANSFER') & (df['amount'] > 200000), 'isFlaggedFraud'] = 1

        logger.info(f"Generated synthetic PaySim data. Fraud rate: {df['isFraud'].mean()*100:.2f}%")
        return df


class TaxFraudDataset(BaseFinancialDataset):
    """
    Synthetic Tax Return Dataset for Fraud Detection.
    """
    @classmethod
    def load_or_generate(cls, n_samples: int = 50000) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Generating {n_samples} synthetic Tax Fraud records...")
        df = cls.generate_synthetic(n_samples)
        
        # Feature Engineering for Tax
        df['income_discrepancy'] = df['reported_income'] - df['actual_income']
        df['deduction_to_income_ratio'] = df['deductions_claimed'] / (df['reported_income'] + 1e-8)
        df['tax_liability_ratio'] = df['tax_owed'] / (df['reported_income'] + 1e-8)
        
        labels = df['is_fraud']
        features = df.drop(columns=['is_fraud', 'taxpayer_id'])
        
        return features, labels

    @classmethod
    def generate_synthetic(cls, n_samples: int = 50000) -> pd.DataFrame:
        np.random.seed(42)
        
        actual_income = np.random.lognormal(mean=11, sigma=0.8, size=n_samples) # ~60k mean
        reported_income = actual_income.copy()
        
        deductions_claimed = reported_income * np.random.uniform(0.05, 0.25, size=n_samples)
        tax_paid = (reported_income - deductions_claimed) * 0.15
        tax_owed = (reported_income - deductions_claimed) * 0.15
        refund_claimed = np.maximum(0, tax_paid - tax_owed)
        
        df = pd.DataFrame({
            'taxpayer_id': [f'T{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
            'filing_year': np.random.choice([2021, 2022, 2023], size=n_samples),
            'reported_income': reported_income,
            'actual_income': actual_income,
            'deductions_claimed': deductions_claimed,
            'tax_paid': tax_paid,
            'tax_owed': tax_owed,
            'refund_claimed': refund_claimed,
            'num_dependents': np.random.poisson(lam=1.5, size=n_samples),
            'filing_status': np.random.choice(['SINGLE', 'MARRIED_JOINT', 'HEAD_HOUSEHOLD'], n_samples),
            'occupation_code': np.random.randint(10, 99, size=n_samples),
            'region_code': np.random.randint(1, 10, size=n_samples),
            'previous_audit_flag': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
            'is_fraud': 0
        })
        
        # Inject Fraud Patterns (~2%)
        target_fraud = int(n_samples * 0.02)
        fraud_idx = np.random.choice(df.index, target_fraud, replace=False)
        
        for idx in fraud_idx:
            pattern = np.random.choice(['underreport', 'excess_deduct', 'ghost_refund'])
            if pattern == 'underreport':
                df.loc[idx, 'reported_income'] *= np.random.uniform(0.3, 0.7)
            elif pattern == 'excess_deduct':
                df.loc[idx, 'deductions_claimed'] = df.loc[idx, 'reported_income'] * np.random.uniform(0.5, 0.9)
            elif pattern == 'ghost_refund':
                df.loc[idx, 'refund_claimed'] = np.random.uniform(10000, 50000)
                
            df.loc[idx, 'is_fraud'] = 1

        logger.info(f"Generated synthetic Tax Fraud data. Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        return df


class MultiDatasetLoader:
    """
    Combines and orchestrates data loading across multiple financial datasets.
    Provides unified data loaders for PyTorch.
    """
    def __init__(self, batch_size: int = 1024):
        self.batch_size = batch_size
        self.preprocessor = DataPreprocessor(scaling_method='standard', apply_smote=False)

    def prepare_data(self, 
                     dataset_type: str = 'paysim', 
                     test_size: float = 0.2, 
                     val_size: float = 0.1) -> Dict[str, DataLoader]:
        """
        Loads, preprocesses, splits, and wraps data in DataLoaders.
        """
        if dataset_type == 'paysim':
            X, y = PaySimDataset.load_or_generate()
        elif dataset_type == 'tax':
            X, y = TaxFraudDataset.load_or_generate()
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # Train / Temp (Val + Test) Split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), stratify=y, random_state=42)
        
        # Val / Test Split
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1-val_ratio), stratify=y_temp, random_state=42)

        # Preprocessing
        logger.info("Preprocessing train data...")
        X_train_processed, y_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)

        # PyTorch Datasets
        train_ds = BaseFinancialDataset(X_train_processed, y_train_processed)
        val_ds = BaseFinancialDataset(X_val_processed, y_val)
        test_ds = BaseFinancialDataset(X_test_processed, y_test)

        # Handle class weighting for imbalanced sampler if needed
        class_sample_count = np.array([len(np.where(y_train_processed == t)[0]) for t in np.unique(y_train_processed)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train_processed])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        # Optional: Use WeightedRandomSampler
        # sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        # For large datasets, standard dataloader with loss weighting might be better
        
        loaders = {
            'train': DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        }
        
        logger.info(f"DataLoaders ready. Train batches: {len(loaders['train'])}")
        return loaders
