# BrownBioTech ADMET Deep-Learning Improvement - Iteration 14/100

## File 1: `brownbiotech/models/admet_multitask.py`
```python
"""
Multi-task ADMET prediction architecture.
Shared encoder with task-specific heads for simultaneous property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Try to import rdkit for molecule handling
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@dataclass
class ADMETTaskConfig:
    """Configuration for a single ADMET prediction task."""
    name: str
    task_type: str  # "classification" or "regression"
    output_dim: int = 1
    class_weights: Optional[List[float]] = None
    loss_weight: float = 1.0
    threshold: float = 0.5  # For classification


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task ADMET model."""
    input_dim: int = 1024
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.3
    tasks: List[ADMETTaskConfig] = field(default_factory=list)
    share_bottom: bool = True
    task_specific_layers: int = 1


class GatedMultiTaskLayer(nn.Module):
    """Gated mechanism for balancing shared and task-specific features."""
    
    def __init__(self, shared_dim: int, task_dim: int, num_tasks: int):
        super().__init__()
        self.gate = nn.Linear(shared_dim + task_dim, num_tasks)
        self.num_tasks = num_tasks
        
    def forward(
        self, 
        shared_features: torch.Tensor, 
        task_features: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generate gated combinations of shared and task-specific features.
        
        Args:
            shared_features: [batch_size, shared_dim]
            task_features: [batch_size, num_tasks, task_dim]
            
        Returns:
            List of [batch_size, shared_dim] tensors, one per task
        """
        batch_size = shared_features.size(0)
        
        # Expand shared features for each task
        shared_expanded = shared_features.unsqueeze(1).expand(-1, self.num_tasks, -1)
        
        # Compute gates
        combined = torch.cat([shared_expanded, task_features], dim=-1)
        gates = F.softmax(self.gate(combined), dim=-1)  # [batch, num_tasks, num_tasks]
        
        # Apply gates to create task-specific outputs
        outputs = []
        for t in range(self.num_tasks):
            gate_weights = gates[:, t, :].unsqueeze(-1)  # [batch, num_tasks, 1]
            weighted = (shared_expanded * gate_weights).sum(dim=1)  # [batch, shared_dim]
            outputs.append(weighted + task_features[:, t, :])
            
        return outputs


class SharedEncoder(nn.Module):
    """Shared bottom layers for feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features from input."""
        return self.network(x)


class TaskHead(nn.Module):
    """Task-specific prediction head."""
    
    def __init__(
        self, 
        input_dim: int, 
        config: ADMETTaskConfig,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.config = config
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, config.output_dim)
        ]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate task predictions."""
        return self.network(x)


class ADMETMultiTaskModel(nn.Module):
    """
    Multi-task ADMET prediction model with shared encoder and task-specific heads.
    
    Supports both classification (binary/multi-class) and regression tasks
    simultaneously, enabling knowledge transfer between related ADMET properties.
    """
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        if not config.tasks:
            raise ValueError("At least one task must be specified")
            
        # Shared encoder
        self.shared_encoder = SharedEncoder(
            config.input_dim, 
            config.hidden_dims, 
            config.dropout
        )
        
        # Task-specific layers
        encoder_output_dim = self.shared_encoder.output_dim
        task_dim = config.hidden_dims[-1] if config.hidden_dims else 256
        
        if config.share_bottom:
            self.task_specific = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(encoder_output_dim, task_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout * 0.5)
                )
                for _ in config.tasks
            ])
            self.gate = GatedMultiTaskLayer(
                encoder_output_dim, task_dim, len(config.tasks)
            )
            head_input_dim = encoder_output_dim + task_dim
        else:
            self.task_specific = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(encoder_output_dim, task_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                )
                for _ in config.tasks
            ])
            self.gate = None
            head_input_dim = task_dim
        
        # Task heads
        self.task_heads = nn.ModuleList([
            TaskHead(head_input_dim, task_config)
            for task_config in config.tasks
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.
        
        Args:
            x: Input features [batch_size, input_dim]
            return_features: If True, also return intermediate features
            
        Returns:
            Dictionary mapping task names to predictions
        """
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Task-specific features
        task_features_list = [layer(shared_features) for layer in self.task_specific]
        task_features = torch.stack(task_features_list, dim=1)  # [batch, num_tasks, task_dim]
        
        # Apply gating if enabled
        if self.gate is not None:
            gated_features = self.gate(shared_features, task_features)
            head_inputs = [
                torch.cat([g, t], dim=-1) 
                for g, t in zip(gated_features, task_features_list)
            ]
        else:
            head_inputs = task_features_list
            
        # Task predictions
        outputs = {}
        for i, task_config in enumerate(self.config.tasks):
            outputs[task_config.name] = self.task_heads[i](head_inputs[i])
            
        if return_features:
            outputs["_shared_features"] = shared_features
            outputs["_task_features"] = task_features
            
        return outputs
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.
        
        Args:
            predictions: Model predictions per task
            targets: Ground truth labels per task
            task_masks: Optional masks for missing labels
            
        Returns:
            Tuple of (total_loss, per_task_losses)
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        task_losses = {}
        
        for task_config in self.config.tasks:
            name = task_config.name
            if name not in predictions or name not in targets:
                continue
                
            pred = predictions[name]
            target = targets[name]
            mask = task_masks.get(name, None) if task_masks else None
            
            # Compute task-specific loss
            if task_config.task_type == "classification":
                if task_config.output_dim == 1:
                    # Binary classification
                    loss = F.binary_cross_entropy_with_logits(
                        pred.squeeze(-1), 
                        target.float(),
                        reduction='none'
                    )
                else:
                    # Multi-class classification
                    loss = F.cross_entropy(
                        pred, 
                        target.long(),
                        reduction='none'
                    )
            else:
                # Regression
                loss = F.mse_loss(pred.squeeze(-1), target.float(), reduction='none')
            
            # Apply mask for missing values
            if mask is not None:
                loss = loss * mask
                loss = loss.sum() / mask.sum().clamp(min=1)
            else:
                loss = loss.mean()
                
            # Apply task weight
            weighted_loss = loss * task_config.loss_weight
            total_loss = total_loss + weighted_loss
            task_losses[name] = loss.item()
            
        return total_loss, task_losses
    
    def predict_proba(
        self, 
        x: torch.Tensor,
        task_name: str
    ) -> torch.Tensor:
        """
        Get probability predictions for a classification task.
        
        Args:
            x: Input features
            task_name: Name of the classification task
            
        Returns:
            Probabilities [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            pred = predictions[task_name]
            
            task_config = next(
                t for t in self.config.tasks if t.name == task_name
            )
            
            if task_config.task_type != "classification":
                raise ValueError(f"Task {task_name} is not a classification task")
                
            if task_config.output_dim == 1:
                return torch.sigmoid(pred.squeeze(-1))
            else:
                return F.softmax(pred, dim=-1)


def create_default_admet_config() -> MultiTaskConfig:
    """Create default configuration for standard ADMET tasks."""
    tasks = [
        ADMETTaskConfig(
            name="absorption",
            task_type="classification",
            output_dim=1,
            loss_weight=1.0
        ),
        ADMETTaskConfig(
            name="distribution_vd",
            task_type="regression",
            output_dim=1,
            loss_weight=0.8
        ),
        ADMETTaskConfig(
            name="metabolism_stability",
            task_type="classification",
            output_dim=1,
            loss_weight=1.2
        ),
        ADMETTaskConfig(
            name="excretion_clearance",
            task_type="regression",
            output_dim=1,
            loss_weight=0.7
        ),
        ADMETTaskConfig(
            name="toxicity_herg",
            task_type="classification",
            output_dim=1,
            loss_weight=1.5  # Higher weight for safety-critical
        ),
        ADMETTaskConfig(
            name="toxicity_ames",
            task_type="classification",
            output_dim=1,
            loss_weight=1.5
        ),
        ADMETTaskConfig(
            name="cyp_inhibition",
            task_type="classification",
            output_dim=5,  # CYP2C9, CYP2C19, CYP2D6, CYP3A4, CYP1A2
            loss_weight=1.0
        ),
    ]
    
    return MultiTaskConfig(
        input_dim=1024,
        hidden_dims=[512, 256],
        dropout=0.3,
        tasks=tasks,
        share_bottom=True
    )


if __name__ == "__main__":
    # Test the multi-task model
    config = create_default_admet_config()
    model = ADMETMultiTaskModel(config)
    
    # Dummy input
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    
    # Forward pass
    outputs = model(x, return_features=True)
    
    print("Multi-task ADMET Model Test")
    print("=" * 50)
    for name, pred in outputs.items():
        if not name.startswith("_"):
            print(f"{name}: {pred.shape}")
    
    # Test loss computation
    targets = {}
    task_masks = {}
    for task in config.tasks:
        if task.task_type == "classification":
            if task.output_dim == 1:
                targets[task.name] = torch.randint(0, 2, (batch_size,))
            else:
                targets[task.name] = torch.randint(0, task.output_dim, (batch_size,))
        else:
            targets[task.name] = torch.randn(batch_size)
        # Simulate some missing values
        mask = torch.ones(batch_size)
        mask[::5] = 0  # 20% missing
        task_masks[task.name] = mask
    
    total_loss, task_losses = model.compute_loss(outputs, targets, task_masks)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Per-task losses: {task_losses}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
```

## File 2: `brownbiotech/models/admet_transfer.py`
```python
"""
Transfer learning pipeline for ADMET prediction.
Enables pretraining on large datasets and fine-tuning on target datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging

from .admet_multitask import (
    ADMETMultiTaskModel, 
    MultiTaskConfig, 
    ADMETTaskConfig
)

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    # Pretraining
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    
    # Fine-tuning
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4
    finetune_weight_decay: float = 1e-4
    
    # Strategy
    freeze_strategy: str = "gradual"  # "none", "full", "gradual", "layerwise"
    unfreeze_schedule: List[int] = field(default_factory=lambda: [5, 10, 15])
    differential_lr_ratio: float = 0.1  # Frozen layers get this * learning_rate
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_ema: bool = False
    ema_decay: float = 0.999
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
                
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class MixupAugmentation:
    """Mixup data augmentation for regularization."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(
        self, 
        x: torch.Tensor, 
        y: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, float]:
        """
        Apply mixup augmentation.
        
        Args:
            x: Input features [batch_size, ...]
            y: Target dictionary
            
        Returns:
            Tuple of (mixed_x, mixed_y, lambda, original_lambda)
        """
        if self.alpha <= 0:
            return x, y, torch.ones(x.size(0)), 1.0
            
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = {}
        for key, val in y.items():
            if val.dtype in [torch.float32, torch.float64]:
                mixed_y[key] = lam * val + (1 - lam) * val[index]
            else:
                mixed_y[key] = val  # Don't mix integer labels
                
        return mixed_x, mixed_y, index, lam


class TransferLearningPipeline:
    """
    Pipeline for transfer learning on ADMET prediction tasks.
    
    Supports:
    - Pretraining on large source datasets (e.g., ChEMBL, Tox21)
    - Fine-tuning on smaller target datasets
    - Multiple freezing strategies
    - EMA for stable predictions
    - Mixup augmentation
    """
    
    def __init__(
        self,
        model: ADMETMultiTaskModel,
        config: TransferConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model.to(device or torch.device("cpu"))
        self.config = config
        self.device = device or torch.device("cpu")
        
        self.ema: Optional[EMAModel] = None
        if config.use_ema:
            self.ema = EMAModel(model, config.ema_decay)
            
        self.mixup = MixupAugmentation(config.mixup_alpha)
        self.history: Dict[str, List[float]] = {
            "pretrain_loss": [],
            "pretrain_task_losses": [],
            "finetune_loss": [],
            "finetune_val_loss": [],
            "finetune_task_losses": [],
        }
        
    def _get_parameter_groups(
        self, 
        differential: bool = False
    ) -> List[Dict[str, Union[float, List[nn.Parameter]]]]:
        """Create parameter groups with optional differential learning rates."""
        if not differential:
            return [{"params": self.model.parameters()}]
            
        shared_params = []
        task_params = []
        
        for name, param in self.model.named_parameters():
            if "shared_encoder" in name:
                shared_params.append(param)
            else:
                task_params.append(param)
                
        return [
            {
                "params": shared_params,
                "lr": self.config.finetune_lr * self.config.differential_lr_ratio
            },
            {
                "params": task_params,
                "lr": self.config.finetune_lr
            }
        ]
    
    def _freeze_layers(self, strategy: str, epoch: int = 0) -> None:
        """Apply freezing strategy based on current epoch."""
        if strategy == "none":
            for param in self.model.parameters():
                param.requires_grad = True
                
        elif strategy == "full":
            # Freeze shared encoder completely
            for name, param in self.model.named_parameters():
                if "shared_encoder" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        elif strategy == "gradual":
            # Gradually unfreeze based on schedule
            schedule = sorted(self.config.unfreeze_schedule)
            
            if epoch < schedule[0]:
                # Freeze all shared layers
                for name, param in self.model.named_parameters():
                    if "shared_encoder" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                # Unfreeze progressively
                shared_layers = list(self.model.shared_encoder.network.children())
                num_to_unfreeze = 0
                
                for i, unfreeze_epoch in enumerate(schedule):
                    if epoch >= unfreeze_epoch:
                        num_to_unfreeze = i + 1
                        
                # Unfreeze from the end (closest to task heads)
                for idx, layer in enumerate(shared_layers):
                    layer_idx = len(shared_layers) - 1 - idx
                    should_unfreeze = layer_idx < num_to_unfreeze * 2
                    
                    for param in layer.parameters():
                        param.requires_grad = should_unfreeze
                        
                # Task layers always unfrozen
                for name, param in self.model.named_parameters():
                    if "shared_encoder" not in name:
                        param.requires_grad = True
                        
        elif strategy == "layerwise":
            # Unfreeze one layer per schedule point
            shared_layers = list(self.model.shared_encoder.network.children())
            num_layers = len(shared_layers)
            layers_per_step = max(1, num_layers // (len(self.config.unfreeze_schedule) + 1))
            
            for idx, layer in enumerate(shared_layers):
                # Layers closer to input are frozen longer
                threshold = epoch * layers_per_step
                param.requires_grad = idx >= (num_layers - threshold)
                
        # Log frozen status
        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.model.parameters())
        logger.debug(f"Epoch {epoch}: Frozen {frozen_count}/{total_count} parameters")
        
    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Pretrain model on source dataset.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            callback: Optional callback(epoch, loss)
            
        Returns:
            Training history
        """
        logger.info(f"Starting pretraining for {self.config.pretrain_epochs} epochs")
        
        # Unfreeze all for pretraining
        for param in self.model.parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=self.config.pretrain_weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.pretrain_epochs
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.pretrain_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                x, targets, masks = self._unpack_batch(batch)
                x = x.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                if masks:
                    masks = {k: v.to(self.device) for k, v in masks.items()}
                
                # Apply mixup
                if self.config.mixup_alpha > 0 and self.model.training:
                    x, targets, _, _ = self.mixup(x, targets)
                
                optimizer.zero_grad()
                predictions = self.model(x)
                loss, _ = self.model.compute_loss(predictions, targets, masks)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                if self.ema:
                    self.ema.update()
                    
                epoch_loss += loss.item()
                num_batches += 1
                
            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            self.history["pretrain_loss"].append(avg_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            if callback:
                callback(epoch, avg_loss)
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Pretrain epoch {epoch+1}: loss={avg_loss:.4f}")
                
        return self.history
    
    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callback: Optional[Callable[[int, float, float], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Fine-tune model on target dataset with transfer learning.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
            callback: Optional callback(epoch, train_loss, val_loss)
            
        Returns:
            Training history
        """
        logger.info(f"Starting fine-tuning for {self.config.finetune_epochs} epochs")
        logger.info(f"Freeze strategy: {self.config.freeze_strategy}")
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.config.finetune_epochs):
            # Apply freezing strategy
            self._freeze_layers(self.config.freeze_strategy, epoch)
            
            # Create optimizer with current trainable parameters
            differential = self.config.freeze_strategy in ["gradual", "layerwise"]
            param_groups = self._get_parameter_groups(differential)
            
            optimizer = optim.AdamW(
                param_groups,
                lr=self.config.finetune_lr,
                weight_decay=self.config.finetune_weight_decay
            )
            
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                x, targets, masks = self._unpack_batch(batch)
                x = x.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                if masks:
                    masks = {k: v.to(self.device) for k, v in masks.items()}
                
                optimizer.zero_grad()
                predictions = self.model(x)
                loss, _ = self.model.compute_loss(predictions, targets, masks)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if self.ema:
                    self.ema.update()
                    
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.history["finetune_loss"].append(avg_train_loss)
            
            # Validation
            val_loss = float('inf')
            if val_loader is not None:
                if self.ema:
                    self.ema.apply_shadow()
                val_loss = self._evaluate(val_loader)
                if self.ema:
                    self.ema.restore()
                    
                self.history["finetune_val_loss"].append(val_loss)
                
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.history["finetune_val_loss"].append(float('nan'))
                
            if callback:
                callback(epoch, avg_train_loss, val_loss)
                
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Finetune epoch {epoch+1}: "
                    f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        # Unfreeze all for inference
        for param in self.model.parameters():
            param.requires_grad = True
            
        return self.history
    
    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on data loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                x, targets, masks = self._unpack_batch(batch)
                x = x.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                if masks:
                    masks = {k: v.to(self.device) for k, v in masks.items()}
                
                predictions = self.model(x)
                loss, _ = self.model.compute_loss(predictions, targets, masks)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(num_batches, 1)
    
    def _unpack_batch(
        self, 
        batch: Tuple
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Unpack batch from data loader.
        Supports multiple batch formats.
        """
        if len(batch) == 2:
            return batch[0], batch[1], None
        elif len(batch) == 3:
            return batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "config": self.config,
        }
        
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model and training state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.history = state.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path}")


def create_transfer_pipeline(
    source_tasks: List[ADMETTaskConfig],
    target_tasks: List[ADMETTaskConfig],
    input_dim: int = 1024,
    device: Optional[torch.device] = None
) -> TransferLearningPipeline:
    """
    Create a transfer learning pipeline with source and target tasks.
    
    Args:
        source_tasks: Tasks for pretraining
        target_tasks: Tasks for fine-tuning (subset or extension of source)
        input_dim: Input feature dimension
        device: Torch device
        
    Returns:
        Configured TransferLearningPipeline
    """
    # Model includes all tasks (source + target)
    all_tasks = source_tasks + [t for t in target_tasks if t.name not in [s.name for s in source_tasks]]
    
    model_config = MultiTaskConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256],
        dropout=0.3,
        tasks=all_tasks,
        share_bottom=True
    )
    
    model = ADMETMultiTaskModel(model_config)
    transfer_config = TransferConfig()
    
    return TransferLearningPipeline(model, transfer_config, device)


if __name__ == "__main__":
    # Test transfer learning pipeline
    from .admet_multitask import create_default_admet_config
    
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    config = create_default_admet_config()
    model = ADMETMultiTaskModel(config)
    
    # Create pipeline
    pipeline = TransferLearningPipeline(
        model, 
        TransferConfig(
            pretrain_epochs=5,
            finetune_epochs=5,
            patience=3
        )
    )
    
    # Create dummy data
    class DummyDataset(Dataset):
        def __init__(self, size=100, input_dim=1024, tasks=None):
            self.x = torch.randn(size, input_dim)
            self.tasks = tasks or []
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, idx):
            targets = {}
            masks = {}
            for task in self.tasks:
                if task.task_type == "classification":
                    targets[task.name] = torch.randint(0, 2, (1,)).float()
                else:
                    targets[task.name] = torch.randn(1)
                masks[task.name] = torch.tensor(1.0)
            return self.x[idx], targets, masks
    
    train_dataset = DummyDataset(64, config.input_dim, config.tasks)
    val_dataset = DummyDataset(16, config.input_dim, config.tasks)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Pretrain
    print("Pretraining...")
    history = pipeline.pretrain(train_loader, val_loader)
    print(f"Pretrain losses: {history['pretrain_loss']}")
    
    # Finetune
    print("\nFine-tuning...")
    history = pipeline.finetune(train_loader, val_loader)
    print(f"Finetune losses: {history['finetune_loss']}")
    print(f"Val losses: {history['finetune_val_loss']}")
```

## File 3: `brownbiotech/models/explainability.py`
```python
"""
Attention saliency and explainability module for ADMET predictions.
Provides interpretability through gradient-based and attention-based methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict

# Try importing matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class SaliencyResult:
    """Container for saliency/explainability results."""
    task_name: str
    saliency_map: np.ndarray  # [input_dim] or [seq_len, input_dim]
    attribution_scores: Dict[str, float]
    top_features: List[Tuple[int, float]]
    prediction: float
    prediction_proba: Optional[float] = None


class GradientSaliency:
    """
    Gradient-based saliency maps for input feature attribution.
    
    Computes gradients of output with respect to input to identify
    important features for each prediction.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
    def compute_saliency(
        self,
        x: torch.Tensor,
        task_name: str,
        target_class: int = 1,
        abs_gradient: bool = True
    ) -> SaliencyResult:
        """
        Compute saliency map using vanilla gradients.
        
        Args:
            x: Input tensor [1, input_dim]
            task_name: Name of the task to explain
            target_class: Target class for classification (0 or 1 for binary)
            abs_gradient: Whether to take absolute value of gradients
            
        Returns:
            SaliencyResult with attribution information
        """
        x = x.clone().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        task_output = outputs[task_name]
        
        # Get prediction
        if task_output.shape[-1] == 1:
            prediction = torch.sigmoid(task_output).item()
            target = torch.tensor([target_class], dtype=torch.float, device=x.device)
            loss = F.binary_cross_entropy(task_output.squeeze(-1), target)
        else:
            prediction = F.softmax(task_output, dim=-1)[0, target_class].item()
            loss = task_output[0, target_class]
            
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Get gradients
        gradients = x.grad.detach().cpu().numpy()
        if abs_gradient:
            gradients = np.abs(gradients)
            
        gradients = gradients.squeeze()
        
        # Normalize
        grad_sum = gradients.sum()
        if grad_sum > 0:
            normalized = gradients / grad_sum
        else:
            normalized = gradients
            
        # Get top features
        top_indices = np.argsort(normalized)[-10:][::-1]
        top_features = [(int(idx), float(normalized[idx])) for idx in top_indices]
        
        # Create attribution summary
        attribution_scores = {
            "max": float(gradients.max()),
            "mean": float(gradients.mean()),
            "std": float(gradients.std()),
            "sparsity": float(np.sum(gradients == 0) / len(gradients))
        }
        
        return SaliencyResult(
            task_name=task_name,
            saliency_map=gradients,
            attribution_scores=attribution_scores,
            top_features=top_features,
            prediction=prediction
        )
    
    def compute_integrated_gradients(
        self,
        x: torch.Tensor,
        task_name: str,
        target_class: int = 1,
        num_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> SaliencyResult:
        """
        Compute integrated gradients for smoother attributions.
        
        Args:
            x: Input tensor [1, input_dim]
            task_name: Name of the task to explain
            target_class: Target class for classification
            num_steps: Number of integration steps
            baseline: Baseline input (default: zeros)
            
        Returns:
            SaliencyResult with integrated gradient attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(x)
            
        # Generate interpolation path
        alphas = torch.linspace(0, 1, num_steps, device=x.device)
        
        integrated_grads = torch.zeros_like(x)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().requires_grad_(True)
            
            outputs = self.model(interpolated)
            task_output = outputs[task_name]
            
            if task_output.shape[-1] == 1:
                target = torch.tensor([target_class], dtype=torch.float, device=x.device)
                loss = F.binary_cross_entropy(task_output.squeeze(-1), target)
            else:
                loss = task_output[0, target_class]
                
            self.model.zero_grad()
            loss.backward()
            
            integrated_grads += interpolated.grad.detach()
            
        # Average and scale
        integrated_grads = integrated_grads / num_steps
        integrated_grads = (x - baseline) * integrated_grads
        integrated_grads = integrated_grads.detach().cpu().numpy().squeeze()
        integrated_grads = np.abs(integrated_grads)
        
        # Normalize
        grad_sum = integrated_grads.sum()
        if grad_sum > 0:
            normalized = integrated_grads / grad_sum
        else:
            normalized = integrated_grads
            
        # Get top features
        top_indices = np.argsort(normalized)[-10:][::-1]
        top_features = [(int(idx), float(normalized[idx])) for idx in top_indices]
        
        # Get final prediction
        with torch.no_grad():
            outputs = self.model(x)
            task_output = outputs[task_name]
            if task_output.shape[-1] == 1:
                prediction = torch.sigmoid(task_output).item()
            else:
                prediction = F.softmax(task_output, dim=-1)[0, target_class].item()
        
        return SaliencyResult(
            task_name=task_name,
            saliency_map=integrated_grads,
            attribution_scores={
                "max": float(integrated_grads.max()),
                "mean": float(integrated_grads.mean()),
                "std": float(integrated_grads.std()),
                "sparsity": float(np.sum(integrated_grads == 0) / len(integrated_grads))
            },
            top_features=top_features,
            prediction=prediction
        )


class AttentionExplainer:
    """
    Attention-based explanation module.
    
    Extracts and visualizes attention weights from models with
    attention mechanisms.
    """
    
    def __init__(self, model: nn.Module, attention_layer_name: str = "attention"):
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        self.handles = []
        
        def get_attention_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach()
            return hook
            
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name.lower():
                handle = module.register_forward_hook(get_attention_hook(name))
                self.handles.append(handle)
                
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def get_attention_map(
        self,
        x: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for an input.
        
        Args:
            x: Input tensor
            layer_name: Specific layer name (default: first attention layer)
            
        Returns:
            Attention weights tensor or None if no attention found
        """
        self.attention_weights = {}
        
        with torch.no_grad():
            self.model(x)
            
        if not self.attention_weights:
            return None
            
        if layer_name and layer_name in self.attention_weights:
            return self.attention_weights[layer_name]
        else:
            return list(self.attention_weights.values())[0]


class FeatureImportanceAggregator:
    """
    Aggregate feature importance across multiple predictions.
    
    Provides global explanations by aggregating local explanations
    from multiple samples.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.accumulated_importance: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
        self.predictions: Dict[str, List[float]] = defaultdict(list)
        
    def add_result(self, result: SaliencyResult) -> None:
        """Add a saliency result to the aggregation."""
        self.accumulated_importance[result.task_name] = np.vstack([
            self.accumulated_importance[result.task_name],
            result.saliency_map
        ]) if len(self.accumulated_importance[result.task_name]) > 0 else result.saliency_map
        self.predictions[result.task_name].append(result.prediction)
        
    def get_global_importance(
        self, 
        task_name: str,
        method: str = "mean_abs"
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """
        Get global feature importance for a task.
        
        Args:
            task_name: Name of the task
            method: Aggregation method ("mean_abs", "mean", "frequency")
            
        Returns:
            Tuple of (importance_array, sorted_feature_importance)
        """
        if task_name not in self.accumulated_importance:
            raise ValueError(f"No results for task {task_name}")
            
        importance = self.accumulated_importance[task_name]
        
        if method == "mean_abs":
            global_importance = np.mean(np.abs(importance), axis=0)
        elif method == "mean":
            global_importance = np.mean(importance, axis=0)
        elif method == "frequency":
            # Count how often each feature is in top-k
            k = min(10, importance.shape[1])
            top_k_indices = np.argsort(np.abs(importance), axis=1)[:, -k:]
            global_importance = np.zeros(importance.shape[1])
            for indices in top_k_indices:
                global_importance[indices] += 1
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Normalize
        if global_importance.sum() > 0:
            global_importance = global_importance / global_importance.sum()
            
        # Sort
        sorted_indices = np.argsort(global_importance)[::-1]
        sorted_importance = [(int(idx), float(global_importance[idx])) for idx in sorted_indices]
        
        return global_importance, sorted_importance
    
    def get_feature_name(self, index: int) -> str:
        """Get feature name for an index."""
        if self.feature_names and index < len(self.feature_names):
            return self.feature_names[index]
        return f"feature_{index}"
    
    def summary(self, task_name: str, top_k: int = 10) -> str:
        """Generate human-readable summary of feature importance."""
        _, sorted_importance = self.get_global_importance(task_name)
        
        lines = [f"\nFeature Importance Summary for {task_name}:"]
        lines.append("=" * 50)
        
        for rank, (idx, score) in enumerate(sorted_importance[:top_k], 1):
            name = self.get_feature_name(idx)
            lines.append(f"{rank:2d}. {name:30s} {score:.4f}")
            
        return "\n".join(lines)


def visualize_saliency(
    result: SaliencyResult,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: Optional[str] = None
) -> Optional[object]:
    """
    Visualize saliency results as a bar plot.
    
    Args:
        result: SaliencyResult to visualize
        feature_names: Optional feature names
        top_k: Number of top features to show
        title: Plot title
        
    Returns:
        Matplotlib figure or None if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get top features
    top_features = result.top_features[:top_k]
    indices = [f[0] for f in reversed(top_features)]
    scores = [f[1] for f in reversed(top_features)]
    
    # Get labels
    if feature_names:
        labels = [feature_names[idx] if idx < len(feature_names) else f"feat_{idx}" 
                  for idx in indices]
    else:
        labels = [f"feature_{idx}" for idx in indices]
    
    # Create bar plot
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(scores)))
    bars = ax.barh(range(len(labels)), scores, color=colors)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Importance Score')
    
    if title:
        ax.set_title(title)
    else:
        pred_str = f"{result.prediction:.3f}"
        ax.set_title(
            f"Feature Importance: {result.task_name}\n"
            f"Prediction: {pred_str}"
        )
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import torch
    from .admet_multitask import ADMETMultiTaskModel, create_default_admet_config
    
    # Create model
    config = create_default_admet_config()
    model = ADMETMultiTaskModel(config)
    model.eval()
    
    # Create explainer
    explainer = GradientSaliency(model)
    
    # Test on dummy input
    x = torch.randn(1, config.input_dim)
    
    print("Gradient Saliency Test")
    print("=" * 50)
    
    # Vanilla gradients
    result = explainer.compute_saliency(x, "toxicity_herg")
    print(f"\nTask: {result.task_name}")
    print(f"Prediction: {result.prediction:.4f}")
    print(f"Attribution scores: {result.attribution_scores}")
    print(f"Top 5 features: {result.top_features[:5]}")
    
    # Integrated gradients
    ig_result = explainer.compute_integrated_gradients(x, "absorption", num_steps=20)
    print(f"\nIntegrated Gradients - Task: {ig_result.task_name}")
    print(f"Prediction: {ig_result.prediction:.4f}")
    print(f"Top 5 features: {ig_result.top_features[:5]}")
    
    # Test aggregation
    aggregator = FeatureImportanceAggregator()
    for _ in range(10):
        x_sample = torch.randn(1, config.input_dim)
        result = explainer.compute_saliency(x_sample, "toxicity_herg")
        aggregator.add_result(result)
        
    print("\n" + aggregator.summary("toxicity_herg", top_k=10))
```

## File 4: `brownbiotech/data/chembl_pretrain.py`
```python
"""
ChEMBL and Tox21 pretraining data loaders.
Provides large-scale datasets for pretraining ADMET models.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import json
import logging
import hashlib

logger = logging.getLogger(__name__)

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Using dummy molecular features.")


@dataclass
class PretrainSample:
    """Single sample for pretraining."""
    smiles: str
    features: np.ndarray
    labels: Dict[str, float]
    mask: Dict[str, float]
    source: str  # "chembl" or "tox21"


class MolecularFeaturizer:
    """
    Convert SMILES to molecular features.
    
    Generates a fixed-size feature vector from molecular structure
    using RDKit descriptors and fingerprints.
    """
    
    def __init__(self, fp_size: int = 1024, include_descriptors: bool = True):
        self.fp_size = fp_size
        self.include_descriptors = include_descriptors
        self.descriptor_names = [
            "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
            "FractionCSP3", "HeavyAtomCount", "RingCount"
        ]
        
    def featurize(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to feature vector.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector or None if invalid SMILES
        """
        if not HAS_RDKIT:
            # Return dummy features for testing
            return np.random.randn(self.fp_size).astype(np.float32)
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        features = []
        
        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=self.fp_size
        )
        features.extend(fp.ToBitString().encode() if False else list(fp))
        
        # Molecular descriptors
        if self.include_descriptors:
            desc_funcs = {
                "MolWt": Descriptors.MolWt,
                "LogP": Descriptors.MolLogP,
                "TPSA": Descriptors.TPSA,
                "NumHDonors": Descriptors.NumHDonors,
                "NumHAcceptors": Descriptors.NumHAcceptors,
                "NumRotatableBonds": Descriptors.NumRotatableBonds,
                "NumAromaticRings": Descriptors.NumAromaticRings,
                "NumAliphaticRings": Descriptors.NumAliphaticRings,
                "FractionCSP3": Descriptors.FractionCSP3,
                "HeavyAtomCount": Descriptors.HeavyAtomCount,
                "RingCount": Descriptors.RingCount,
            }
            
            # Normalize descriptors
            descriptor_values = []
            for name, func in desc_funcs.items():
                try:
                    val = func(mol)
                except:
                    val = 0.0
                descriptor_values.append(val)
                
            # Simple normalization (would use precomputed stats in production)
            descriptor_values = np.array(descriptor_values)
            descriptor_values = (descriptor_values - descriptor_values.mean()) / \
                               (descriptor_values.std() + 1e-8)
            features.extend(descriptor_values.tolist())
            
        return np.array(features, dtype=np.float32)
    
    @property
    def output_dim(self) -> int:
        """Get output feature dimension."""
        dim = self.fp_size
        if self.include_descriptors:
            dim += len(self.descriptor_names)
        return dim


class ChEMBLPretrainDataset(Dataset):
    """
    Dataset for ChEMBL-based pretraining.
    
    Loads preprocessed ChEMBL data with ADMET-related properties
    for pretraining the shared encoder.
    """
    
    # Default ADMET tasks available in ChEMBL
    CHEMBL_TASKS = [
        ("clearance_microsomal", "regression"),
        ("clearance_hepatic", "regression"),
        ("logd", "regression"),
        ("pampa_permeability", "regression"),
        ("solubility", "regression"),
        ("cyp2c9_inhibition", "classification"),
        ("cyp2d6_inhibition", "classification"),
        ("cyp3a4_inhibition", "classification"),
        ("cyp2c19_inhibition", "classification"),
        ("cyp1a2_inhibition", "classification"),
        ("ppb", "regression"),  # Plasma protein binding
        ("vdss", "regression"),  # Volume of distribution
    ]
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        featurizer: Optional[MolecularFeaturizer] = None,
        max_samples: Optional[int] = None,
        cache_features: bool = True
    ):
        """
        Initialize ChEMBL pretraining dataset.
        
        Args:
            data_path: Path to preprocessed data file
            featurizer: Molecular featurizer instance
            max_samples: Maximum number of samples to load
            cache_features: Whether to cache computed features
        """
        self.featurizer = featurizer or MolecularFeaturizer()
        self.samples: List[PretrainSample] = []
        self.cache_features = cache_features
        self._feature_cache: Dict[str, np.ndarray] = {}
        
        if data_path and data_path.exists():
            self._load_from_file(data_path, max_samples)
        else:
            logger.warning(
                f"Data path not found: {data_path}. "
                "Using synthetic data for testing."
            )
            self._generate_synthetic_data(max_samples or 1000)
            
    def _load_from_file(self, path: Path, max_samples: Optional[int]) -> None:
        """Load data from preprocessed JSON/CSV file."""
        with open(path, 'r') as f:
            if path.suffix == '.json':
                data = json.load(f)
            else:
                # Simple CSV parsing
                import csv
                reader = csv.DictReader(f)
                data = list(reader)
                
        count = 0
        for item in data:
            if max_samples and count >= max_samples:
                break
                
            smiles = item.get("smiles", "")
            if not smiles:
                continue
                
            features = self._get_features(smiles)
            if features is None:
                continue
                
            labels = {}
            mask = {}
            for task_name, task_type in self.CHEMBL_TASKS:
                val = item.get(task_name)
                if val is not None and val != "":
                    try:
                        labels[task_name] = float(val)
                        mask[task_name] = 1.0
                    except ValueError:
                        mask[task_name] = 0.0
                else:
                    mask[task_name] = 0.0
                    
            if any(m == 1.0 for m in mask.values()):
                self.samples.append(PretrainSample(
                    smiles=smiles,
                    features=features,
                    labels=labels,
                    mask=mask,
                    source="chembl"
                ))
                count += 1
                
        logger.info(f"Loaded {len(self.samples)} samples from {path}")
        
    def _generate_synthetic_data(self, n_samples: int) -> None:
        """Generate synthetic data for testing."""
        np.random.seed(42)
        
        for i in range(n_samples):
            smiles = f"C{'C' * np.random.randint(1, 20)}"
            features = self._get_features(smiles)
            
            labels = {}
            mask = {}
            for task_name, task_type in self.CHEMBL_TASKS:
                if np.random.random() > 0.2:  # 80% label availability
                    if task_type == "classification":
                        labels[task_name] = float(np.random.randint(0, 2))
                    else:
                        labels[task_name] = np.random.randn()
                    mask[task_name] = 1.0
                else:
                    mask[task_name] = 0.0
                    
            self.samples.append(PretrainSample(
                smiles=smiles,
                features=features,
                labels=labels,
                mask=mask,
                source="chembl"
            ))
            
        logger.info(f"Generated {n_samples} synthetic samples")
        
    def _get_features(self, smiles: str) -> Optional[np.ndarray]:
        """Get features with caching."""
        if self.cache_features and smiles in self._feature_cache:
            return self._feature_cache[smiles]
            
        features = self.featurizer.featurize(smiles)
        
        if features is not None and self.cache_features:
            self._feature_cache[smiles] = features
            
        return features
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        features = torch.tensor(sample.features, dtype=torch.float32)
        labels = {k: torch.tensor(v, dtype=torch.float32) for k, v in sample.labels.items()}
        mask = {k: torch.tensor(v, dtype=torch.float32) for k, v in sample.mask.items()}
        
        return features, labels, mask


class Tox21PretrainDataset(Dataset):
    """
    Dataset for Tox21 pretraining.
    
    Tox21 contains toxicity labels across 12 assay endpoints,
    useful for pretraining toxicity prediction heads.
    """
    
    TOX21_TASKS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        featurizer: Optional[MolecularFeaturizer] = None,
        max_samples: Optional[int] = None
    ):
        self.featurizer = featurizer or MolecularFeaturizer()
        self.samples: List[PretrainSample] = []
        self._feature_cache: Dict[str, np.ndarray] = {}
        
        if data_path and data_path.exists():
            self._load_from_file(data_path, max_samples)
        else:
            logger.warning(
                f"Data path not found: {data_path}. "
                "Using synthetic data for testing."
            )
            self._generate_synthetic_data(max_samples or 500)
            
    def _load_from_file(self, path: Path, max_samples: Optional[int]) -> None:
        """Load Tox21 data from file."""
        with open(path, 'r') as f:
            data = json.load(f) if path.suffix == '.json' else list(csv.DictReader(f))
            
        count = 0
        for item in data:
            if max_samples and count >= max_samples:
                break
                
            smiles = item.get("smiles", "")
            features = self.featurizer.featurize(smiles)
            if features is None:
                continue
                
            labels = {}
            mask = {}
            for task in self.TOX21_TASKS:
                val = item.get(task)
                if val is not None and val != "":
                    labels[task] = float(val)
                    mask[task] = 1.0
                else:
                    mask[task] = 0.0
                    
            if any(m == 1.0 for m in mask.values()):
                self.samples.append(PretrainSample(
                    smiles=smiles,
                    features=features,
                    labels=labels,
                    mask=mask,
                    source="tox21"
                ))
                count += 1
                
        logger.info(f"Loaded {len(self.samples)} Tox21 samples")
        
    def _generate_synthetic_data(self, n_samples: int) -> None:
        """Generate synthetic Tox21 data."""
        np.random.seed(123)
        
        for i in range(n_samples):
            smiles = f"C{'C' * np.random.randint(1, 15)}"
            features = self.featurizer.featurize(smiles)
            
            labels = {}
            mask = {}
            for task in self.TOX21_TASKS:
                if np.random.random() > 0.15:  # ~85% label availability in Tox21
                    labels[task] = float(np.random.randint(0, 2))
                    mask[task] = 1.0
                else:
                    mask[task] = 0.0
                    
            self.samples.append(PretrainSample(
                smiles=smiles,
                features=features,
                labels=labels,
                mask=mask,
                source="tox21"
            ))
            
        logger.info(f"Generated {n_samples} synthetic Tox21 samples")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        
        features = torch.tensor(sample.features, dtype=torch.float32)
        labels = {k: torch.tensor(v, dtype=torch.float32) for k, v in sample.labels.items()}
        mask = {k: torch.tensor(v, dtype=torch.float32) for k, v in sample.mask.items()}
        
        return features, labels, mask


class CombinedPretrainDataset(Dataset):
    """
    Combined dataset from multiple sources for pretraining.
    
    Handles different task schemas across datasets by using
    a unified label dictionary with masks for missing values.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        sample_weights: Optional[List[float]] = None
    ):
        self.datasets = datasets
        self.cumulative_sizes = [0]
        
        for ds in datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))
            
        self.total_size = self.cumulative_sizes[-1]
        
        # Compute sampling weights
        if sample_weights is None:
            # Weight by inverse of dataset size for balanced sampling
            sizes = [len(ds) for ds in datasets]
            total = sum(sizes)
            sample_weights = [total / (len(datasets) * s) for s in sizes]
            
        self.sample_weights = sample_weights
        self._build_sampling_table()
        
    def _build_sampling_table(self) -> None:
        """Build table for weighted sampling."""
        self.sampling_table = []
        for ds_idx, (ds, weight) in enumerate(zip(self.datasets, self.sample_weights)):
            n_samples = int(len(ds) * weight)
            self.sampling_table.extend([ds_idx] * n_samples)
            
        np.random.shuffle(self.sampling_table)
        
    def __len__(self) -> int:
        return len(self.sampling_table)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ds_idx = self.sampling_table[idx]
        sample_idx = idx % len(self.datasets[ds_idx])
        return self.datasets[ds_idx][sample_idx]


def create_pretrain_dataloaders(
    chembl_path: Optional[Path] = None,
    tox21_path: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    max_chembl_samples: Optional[int] = None,
    max_tox21_samples: Optional[int] = None,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create pretraining data loaders.
    
    Args:
        chembl_path: Path to ChEMBL data
        tox21_path: Path to Tox21 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_chembl_samples: Max ChEMBL samples
        max_tox21_samples: Max Tox21 samples
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    featurizer = MolecularFeaturizer()
    
    datasets = []
    
    if chembl_path or True:  # Always create for testing
        chembl_ds = ChEMBLPretrainDataset(
            chembl_path, featurizer, max_chembl_samples
        )
        if len(chembl_ds) > 0:
            datasets.append(chembl_ds)
            
    if tox21_path or True:  # Always create for testing
        tox21_ds = Tox21PretrainDataset(
            tox21_path, featurizer, max_tox21_samples
        )
        if len(tox21_ds) > 0:
            datasets.append(tox21_ds)
            
    if not datasets:
        raise ValueError("No valid datasets created")
        
    # Split into train/val
    train_datasets = []
    val_datasets = []
    
    for ds in datasets:
        n_val = int(len(ds) * val_split)
        n_train = len(ds) - n_val
        
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        
    train_combined = CombinedPretrainDataset(train_datasets)
    val_combined = CombinedPretrainDataset(val_datasets)
    
    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_custom_collate
    )
    
    val_loader = DataLoader(
        val_combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_custom_collate
    )
    
    logger.info(
        f"Created pretrain loaders: "
        f"train={len(train_combined)}, val={len(val_combined)}"
    )
    
    return train_loader, val_loader


def _custom_collate(batch):
    """Custom collate function for variable-label batches."""
    features = torch.stack([item[0] for item in batch])
    
    # Collect all label keys
    all_keys = set()
    for _, labels, _ in batch:
        all_keys.update(labels.keys())
        
    # Build label and mask tensors
    labels = {k: [] for k in all_keys}
    masks = {k: [] for k in all_keys}
    
    for _, sample_labels, sample_masks in batch:
        for k in all_keys:
            labels[k].append(sample_labels.get(k, torch.tensor(0.0)))
            masks[k].append(sample_masks.get(k, torch.tensor(0.0)))
            
    labels = {k: torch.stack(v) for k, v in labels.items()}
    masks = {k: torch.stack(v) for k, v in masks.items()}
    
    return features, labels, masks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Pretraining Data Loaders")
    print("=" * 50)
    
    # Test featurizer
    featurizer = MolecularFeaturizer()
    feat = featurizer.featurize("CCO")
    print(f"Feature dimension: {len(feat)}")
    print(f"Expected dimension: {featurizer.output_dim}")
    
    # Test ChEMBL dataset
    chembl_ds = ChEMBLPretrainDataset(max_samples=100)
    print(f"\nChEMBL dataset size: {len(chembl_ds)}")
    x, labels, masks = chembl_ds[0]
    print(f"Feature shape: {x.shape}")
    print(f"Labels: {list(labels.keys())}")
    
    # Test Tox21 dataset
    tox21_ds = Tox21PretrainDataset(max_samples=50)
    print(f"\nTox21 dataset size: {len(tox21_ds)}")
    
    # Test combined dataset
    combined = CombinedPretrainDataset([chembl_ds, tox21_ds])
    print(f"Combined dataset size: {len(combined)}")
    
    # Test data loader
    train_loader, val_loader = create_pretrain_dataloaders(
        batch_size=16,
        max_chembl_samples=100,
        max_tox21_samples=50
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Inspect a batch
    for batch in train_loader:
        x, labels, masks = batch
        print(f"\nBatch features shape: {x.shape}")
        print(f"Batch label keys: {list(labels.keys())}")
        print(f"Sample mask sums: {[(k, v.sum().item()) for k, v in masks.items()]}")
        break
```

## File 5: `brownbiotech/agents/admet_agent.py`
```python
"""
ADMET Agent - Integrated agent for ADMET prediction and optimization.
Combines multi-task model, transfer learning, and explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import json
from datetime import datetime

from ..models.admet_multitask import (
    ADMETMultiTaskModel,
    MultiTaskConfig,
    ADMETTaskConfig,
    create_default_admet_config
)
from ..models.admet_transfer import (
    TransferLearningPipeline,
    TransferConfig,
    create_transfer_pipeline
)
from ..models.explainability import (
    GradientSaliency,
    FeatureImportanceAggregator,
    SaliencyResult,
    visualize_saliency
)
from ..data.chembl_pretrain import (
    MolecularFeaturizer,
    create_pretrain_dataloaders
)

logger = logging.getLogger(__name__)


@dataclass
class ADMETPrediction:
    """Single ADMET prediction result."""
    smiles: str
    task_predictions: Dict[str, float]
    task_probabilities: Dict[str, float]
    risk_assessment: str  # "low", "medium", "high"
    explanations: Optional[Dict[str, SaliencyResult]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class AgentConfig:
    """Configuration for ADMET Agent."""
    model_config: MultiTaskConfig = field(default_factory=create_default_admet_config)
    transfer_config: TransferConfig = field(default_factory=TransferConfig)
    featurizer_fp_size: int = 1024
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    use_explainability: bool = True
    
    # Risk thresholds
    tox_probability_threshold: float = 0.5
    high_risk_threshold: float = 0.7


class ADMETAgent:
    """
    Integrated ADMET prediction and optimization agent.
    
    Provides a unified interface for:
    - Multi-task ADMET prediction
    - Transfer learning from pretraining
    - Explainability and feature attribution
    - Risk assessment and flagging
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        logger.info(f"ADMET Agent initialized on {self.device}")
        
        # Initialize components
        self._init_model()
        self._init_featurizer()
        self._init_explainer()
        self._init_pipeline()
        
        # Load checkpoint if available
        if self.config.checkpoint_path:
            self.load(self.config.checkpoint_path)
            
    def _init_model(self) -> None:
        """Initialize the multi-task model."""
        self.model = ADMETMultiTaskModel(self.config.model_config)
        self.model.to(self.device)
        
    def _init_featurizer(self) -> None:
        """Initialize molecular featurizer."""
        self.featurizer = MolecularFeaturizer(
            fp_size=self.config.featurizer_fp_size
        )
        
        # Update model input dimension if needed
        if self.featurizer.output_dim != self.config.model_config.input_dim:
            logger.warning(
                f"Featurizer output dim ({self.featurizer.output_dim}) != "
                f"model input dim ({self.config.model_config.input_dim}). "
                "Adjusting model config."
            )
            self.config.model_config.input_dim = self.featurizer.output_dim
            self._init_model()
            
    def _init_explainer(self) -> None:
        """Initialize explainability module."""
        if self.config.use_explainability:
            self.explainer = GradientSaliency(self.model)
            self.importance_aggregator = FeatureImportanceAggregator()
        else:
            self.explainer = None
            self.importance_aggregator = None
            
    def _init_pipeline(self) -> None:
        """Initialize transfer learning pipeline."""
        self.pipeline = TransferLearningPipeline(
            self.model,
            self.config.transfer_config,
            self.device
        )
        
    def predict(
        self,
        smiles_list: Union[str, List[str]],
        explain: bool = False
    ) -> Union[ADMETPrediction, List[ADMETPrediction]]:
        """
        Predict ADMET properties for molecules.
        
        Args:
            smiles_list: Single SMILES or list of SMILES
            explain: Whether to compute explanations
            
        Returns:
            ADMETPrediction or list of predictions
        """
        single_input = isinstance(smiles_list, str)
        if single_input:
            smiles_list = [smiles_list]
            
        results = []
        for smiles in smiles_list:
            result = self._predict_single(smiles, explain)
            results.append(result)
            
        return results[0] if single_input else results
        
    def _predict_single(
        self, 
        smiles: str, 
        explain: bool
    ) -> ADMETPrediction:
        """Predict ADMET for a single molecule."""
        # Featurize
        features = self.featurizer.featurize(smiles)
        if features is None:
            raise ValueError(f"Could not featurize SMILES: {smiles}")
            
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            
        # Process outputs
        task_predictions = {}
        task_probabilities = {}
        
        for task_config in self.config.model_config.tasks:
            name = task_config.name
            output = outputs[name].cpu()
            
            if task_config.task_type == "classification":
                if task_config.output_dim == 1:
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob >= task_config.threshold else 0
                else:
                    probs = torch.softmax(output, dim=-1)
                    pred = probs.argmax(dim=-1).item()
                    prob = probs[0, pred].item()
                task_probabilities[name] = prob
            else:
                pred = output.item()
                task_probabilities[name] = pred
                
            task_predictions[name] = pred
            
        # Risk assessment
        risk = self._assess_risk(task_probabilities)
        
        # Explainability
        explanations = None
        if explain and self.explainer:
            explanations = self._explain(x, task_predictions)
            
        return ADMETPrediction(
            smiles=smiles,
            task_predictions=task_predictions,
            task_probabilities=task_probabilities,
            risk_assessment=risk,
            explanations=explanations
        )
        
    def _assess_risk(self, task_probabilities: Dict[str, float]) -> str:
        """Assess overall risk based on predictions."""
        toxicity_tasks = [
            name for name in task_probabilities.keys() 
            if "tox" in name.lower() or "ames" in name.lower() or "herg" in name.lower()
        ]
        
        if not toxicity_tasks:
            return "unknown"
            
        max_tox_prob = max(
            task_probabilities.get(t, 0) for t in toxicity_tasks
        )
        
        if max_tox_prob >= self.config.high_risk_threshold:
            return "high"
        elif max_tox_prob >= self.config.tox_probability_threshold:
            return "medium"
        else:
            return "low"
            
    def _explain(
        self, 
        x: torch.Tensor,
        task_predictions: Dict[str, float]
    ) -> Dict[str, SaliencyResult]:
        """Generate explanations for predictions."""
        explanations = {}
        
        # Focus on toxicity tasks for explanations
        toxicity_tasks = [
            tc for tc in self.config.model_config.tasks
            if "tox" in tc.name.lower() or "ames" in tc.name.lower() or "herg" in tc.name.lower()
        ]
        
        for task_config in toxicity_tasks[:3]:  # Limit to top 3
            try:
                target_class = int(task_predictions.get(task_config.name, 1))
                result = self.explainer.compute_saliency(
                    x, task_config.name, target_class=target_class
                )
                explanations[task_config.name] = result
                
                # Add to aggregator
                if self.importance_aggregator:
                    self.importance_aggregator.add_result(result)
                    
            except Exception as e:
                logger.warning(f"Failed to explain {task_config.name}: {e}")
                
        return explanations
        
    def pretrain(
        self,
        chembl_path: Optional[Path] = None,
        tox21_path: Optional[Path] = None,
        batch_size: int = 64,
        max_epochs: int = 50,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """
        Pretrain on large-scale datasets.
        
        Args:
            chembl_path: Path to ChEMBL data
            tox21_path: Path to Tox21 data
            batch_size: Batch size
            max_epochs: Maximum pretraining epochs
            callback: Optional progress callback
            
        Returns:
            Training history
        """
        logger.info("Starting pretraining phase")
        
        # Create dataloaders
        train_loader, val_loader = create_pretrain_dataloaders(
            chembl_path=chembl_path,
            tox21_path=tox21_path,
            batch_size=batch_size
        )
        
        # Adjust model for pretraining tasks
        self._adjust_model_for_pretraining(train_loader)
        
        # Run pretraining
        self.config.transfer_config.pretrain_epochs = max_epochs
        history = self.pipeline.pretrain(
            train_loader, val_loader, callback
        )
        
        logger.info("Pretraining complete")
        return history
        
    def _adjust_model_for_pretraining(self, train_loader: DataLoader) -> None:
        """Adjust model configuration to match pretraining data."""
        # Get sample batch to determine available tasks
        sample_batch = next(iter(train_loader))
        _, labels, masks = sample_batch
        
        available_tasks = [k for k, v in masks.items() if v.sum() > 0]
        
        logger.info(f"Available pretraining tasks: {available_tasks}")
        
    def finetune(
        self,
        train_data: Union[Dataset, DataLoader],
        val_data: Optional[Union[Dataset, DataLoader]] = None,
        batch_size: int = 32,
        max_epochs: int = 30,
        freeze_strategy: str = "gradual",
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """
        Fine-tune on target dataset.
        
        Args:
            train_data: Training dataset or loader
            val_data: Validation dataset or loader
            batch_size: Batch size
            max_epochs: Maximum fine-tuning epochs
            freeze_strategy: Freezing strategy
            callback: Optional progress callback
            
        Returns:
            Training history
        """
        logger.info(f"Starting fine-tuning with strategy: {freeze_strategy}")
        
        # Create loaders if datasets provided
        if isinstance(train_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            train_loader = train_data
            
        if val_data is not None:
            if isinstance(val_data, Dataset):
                val_loader = DataLoader(val_data, batch_size=batch_size)
            else:
                val_loader = val_data
        else:
            val_loader = None
            
        # Update config
        self.config.transfer_config.finetune_epochs = max_epochs
        self.config.transfer_config.freeze_strategy = freeze_strategy
        
        # Run fine-tuning
        history = self.pipeline.finetune(
            train_loader, val_loader, callback
        )
        
        logger.info("Fine-tuning complete")
        return history
        
    def save(self, path: Union[str, Path]) -> None:
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "version": "14.0"
        }
        
        torch.save(state, path)
        logger.info(f"Saved agent to {path}")
        
    def load(self, path: Union[str, Path]) -> None:
        """Load agent state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded agent from {path}")
        
    def get_feature_importance_summary(
        self, 
        task_name: str, 
        top_k: int = 10
    ) -> str:
        """Get summary of accumulated feature importance."""
        if not self.importance_aggregator:
            return "Explainability not enabled"
        return self.importance_aggregator.summary(task_name, top_k)
        
    def batch_predict_with_filter(
        self,
        smiles_list: List[str],
        max_risk: str = "medium",
        min_absorption_prob: float = 0.5
    ) -> List[ADMETPrediction]:
        """
        Batch predict with filtering criteria.
        
        Args:
            smiles_list: List of SMILES to evaluate
            max_risk: Maximum acceptable risk level
            min_absorption_prob: Minimum absorption probability
            
        Returns:
            Filtered predictions
        """
        risk_order = {"low": 0, "medium": 1, "high": 2}
        max_risk_level = risk_order.get(max_risk, 1)
        
        results = []
        for pred in self.predict(smiles_list):
            # Check risk
            if risk_order.get(pred.risk_assessment, 2) > max_risk_level:
                continue
                
            # Check absorption
            abs_prob = pred.task_probabilities.get("absorption", 0)
            if abs_prob < min_absorption_prob:
                continue
                
            results.append(pred)
            
        return results


def create_default_agent() -> ADMETAgent:
    """Create ADMET agent with default configuration."""
    return ADMETAgent(AgentConfig())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("ADMET Agent Test")
    print("=" * 60)
    
    # Create agent
    agent = create_default_agent()
    
    # Test prediction
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    print("\nPredictions without explanations:")
    for smi in test_smiles:
        pred = agent.predict(smi, explain=False)
        print(f"\n{smi}:")
        print(f"  Risk: {pred.risk_assessment}")
        print(f"  Key predictions: {pred.task_predictions}")
        
    print("\n" + "=" * 60)
    print("\nPrediction with explanations:")
    pred = agent.predict("CC(=O)Oc1ccccc1C(=O)O", explain=True)
    print(f"SMILES: {pred.smiles}")
    print(f"Risk: {pred.risk_assessment}")
    print(f"Predictions: {pred.task_predictions}")
    if pred.explanations:
        for task_name, expl in pred.explanations.items():
            print(f"\n{task_name} explanation:")
            print(f"  Top features: {expl.top_features[:5]}")
            
    # Test batch filtering
    print("\n" + "=" * 60)
    print("\nBatch filtering (low risk, good absorption):")
    filtered = agent.batch_predict_with_filter(
        test_smiles,
        max_risk="low",
        min_absorption_prob=0.3
    )
    print(f"Passed {len(filtered)}/{len(test_smiles)} molecules")
    for pred in filtered:
        print(f"  {pred.smiles}: risk={pred.risk_assessment}")
```

---

## Summary of Improvements (Iteration 14/100)

| Module | Improvement | Key Features |
|--------|-------------|--------------|
| `admet_multitask.py` | Multi-task architecture | Gated sharing, task-specific heads, masked loss for missing labels |
| `admet_transfer.py` | Transfer learning pipeline | 4 freezing strategies, EMA, Mixup augmentation, differential LR |
| `explainability.py` | Attention saliency | Vanilla gradients, integrated gradients, global aggregation |
| `chembl_pretrain.py` | Pretraining data | ChEMBL + Tox21 datasets, RDKit featurizer, weighted sampling |
| `admet_agent.py` | Integrated agent | Unified API, risk assessment, batch filtering, explainability |