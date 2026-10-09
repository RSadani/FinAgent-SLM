import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
import json
from loguru import logger
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in fraud detection."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FraudModelTrainer:
    """Enterprise-grade training pipeline for FinAgent-SLM."""
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        accumulation_steps: int = 1,
        patience: int = 5,
        save_dir: str = "checkpoints"
    ):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Loss functions for multi-task learning
        self.criterion_fraud = FocalLoss()
        self.criterion_type = nn.CrossEntropyLoss()
        self.criterion_risk = nn.MSELoss()
        
        # Mixed precision scaling
        self.scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        fraud_labels = batch['fraud_label'].to(self.device)
        type_labels = batch['fraud_type'].to(self.device)
        risk_scores = batch['risk_score'].to(self.device)
        
        loss_fraud = self.criterion_fraud(outputs['fraud_logits'], fraud_labels)
        loss_type = self.criterion_type(outputs['fraud_type_logits'], type_labels)
        loss_risk = self.criterion_risk(outputs['risk_score'], risk_scores)
        
        # Weighted sum of losses
        return loss_fraud + 0.5 * loss_type + 0.2 * loss_risk

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        best_val_f1 = 0.0
        patience_counter = 0
        
        logger.info(f"Starting training on {self.device} for {epochs} epochs")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                features = batch['features'].to(self.device)
                
                # Mixed precision context
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(features)
                        loss = self._compute_loss(outputs, batch) / self.accumulation_steps
                    self.scaler.scale(loss).backward()
                    
                    if (step + 1) % self.accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(features)
                    loss = self._compute_loss(outputs, batch) / self.accumulation_steps
                    loss.backward()
                    
                    if (step + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps
            
            scheduler.step()
            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['metrics'].append(val_metrics)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping and checkpointing
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self.save_checkpoint(epoch, best_val_f1, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered")
                    break

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['fraud_label'].to(self.device)
                
                outputs = self.model(features)
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
                probs = F.softmax(outputs['fraud_logits'], dim=-1)[:, 1]
                preds = torch.argmax(outputs['fraud_logits'], dim=-1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(dataloader)
        return metrics

    def _compute_metrics(self, y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            'confusion_matrix': cm.tolist()
        }

    def save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric,
            'history': self.history
        }
        
        path = os.path.join(self.save_dir, "latest_checkpoint.pt")
        torch.save(state, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(state, best_path)
            
            with open(os.path.join(self.save_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=4)
