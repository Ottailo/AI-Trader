"""
Model training pipeline for time series prediction.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import structlog

from app.ml.models import LSTMModel, GRUModel
from app.ml.feature_engineering import FeatureEngineer
from app.models import MLModel, ModelStatus
from app.database import AsyncSessionLocal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class ModelTrainer:
    """
    Model training pipeline for time series prediction.
    """
    
    def __init__(
        self,
        model_type: str = "lstm",
        model_params: Optional[Dict] = None,
        training_params: Optional[Dict] = None,
        models_dir: str = "models"
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model ("lstm", "gru", "transformer")
            model_params: Model hyperparameters
            training_params: Training hyperparameters
            models_dir: Directory to save models
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
        
        logger.info("ModelTrainer initialized", 
                   model_type=model_type, 
                   device=str(self.device))
    
    def create_model(self, input_size: int) -> nn.Module:
        """
        Create a new model instance.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch model
        """
        if self.model_type == "lstm":
            self.model = LSTMModel(
                input_size=input_size,
                **self.model_params
            )
        elif self.model_type == "gru":
            self.model = GRUModel(
                input_size=input_size,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        return self.model
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_column: str = "close",
        sequence_length: int = 60,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            sequence_length: Length of input sequences
            train_split: Training data split ratio
            val_split: Validation data split ratio
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, feature_columns)
        """
        logger.info("Preparing data for training", 
                   data_shape=df.shape,
                   target_column=target_column,
                   sequence_length=sequence_length)
        
        # Prepare features
        df_features, feature_cols = self.feature_engineer.prepare_features(
            df, target_column
        )
        
        # Create sequences
        X, y = self.feature_engineer.create_sequences(
            df_features, feature_cols, target_column, sequence_length
        )
        
        logger.info("Created sequences", 
                   X_shape=X.shape, 
                   y_shape=y.shape,
                   feature_count=len(feature_cols))
        
        # Split data
        train_size = int(len(X) * train_split)
        val_size = int(len(X) * val_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Fit scaler on training data
        self.feature_engineer.fit_scaler(X_train)
        
        # Transform all data
        X_train_scaled = self.feature_engineer.transform_features(X_train)
        X_val_scaled = self.feature_engineer.transform_features(X_val)
        X_test_scaled = self.feature_engineer.transform_features(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create data loaders
        batch_size = self.training_params.get("batch_size", 32)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("Data preparation completed",
                   train_size=len(X_train),
                   val_size=len(X_val),
                   test_size=len(X_test),
                   batch_size=batch_size)
        
        return train_loader, val_loader, test_loader, feature_cols
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for the model
            epochs: Number of training epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        logger.info("Starting model training",
                   model_name=model_name,
                   epochs=epochs,
                   learning_rate=learning_rate)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_y.detach().cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_predictions.extend(outputs.detach().cpu().numpy())
                    val_targets.extend(batch_y.detach().cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            
            # Store history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_accuracy"].append(train_rmse)
            self.training_history["val_accuracy"].append(val_rmse)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info("Training progress",
                           epoch=epoch,
                           train_loss=train_loss,
                           val_loss=val_loss,
                           train_rmse=train_rmse,
                           val_rmse=val_rmse)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info("Early stopping triggered",
                           epoch=epoch,
                           best_val_loss=best_val_loss)
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Calculate final metrics
        train_mae = mean_absolute_error(train_targets, train_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        train_r2 = r2_score(train_targets, train_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        results = {
            "model_name": model_name,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_train_rmse": train_rmse,
            "final_val_rmse": val_rmse,
            "final_train_mae": train_mae,
            "final_val_mae": val_mae,
            "final_train_r2": train_r2,
            "final_val_r2": val_r2,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "training_history": self.training_history
        }
        
        logger.info("Training completed", **results)
        return results
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        self.model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                test_predictions.extend(outputs.detach().cpu().numpy())
                test_targets.extend(batch_y.detach().cpu().numpy())
        
        # Calculate metrics
        test_mse = mean_squared_error(test_targets, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        
        # Calculate directional accuracy
        test_targets = np.array(test_targets).flatten()
        test_predictions = np.array(test_predictions).flatten()
        
        directional_accuracy = np.mean(
            np.sign(np.diff(test_targets)) == np.sign(np.diff(test_predictions))
        )
        
        metrics = {
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "directional_accuracy": directional_accuracy
        }
        
        logger.info("Model evaluation completed", **metrics)
        return metrics
    
    def save_model(
        self, 
        model_name: str, 
        version: str = "1.0.0",
        training_results: Optional[Dict] = None,
        evaluation_results: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Save the trained model and related files.
        
        Args:
            model_name: Name for the model
            version: Model version
            training_results: Training results
            evaluation_results: Evaluation results
            
        Returns:
            Dictionary with file paths
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create model directory
        model_dir = self.models_dir / f"{model_name}_v{version}"
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_engineer.scaler, f)
        
        # Save configuration
        config = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "training_params": self.training_params,
            "feature_columns": self.feature_engineer.feature_columns,
            "target_column": "close",  # Default
            "sequence_length": 60,  # Default
            "scaler_type": self.feature_engineer.scaler_type,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "created_at": datetime.now().isoformat()
        }
        
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        file_paths = {
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "config_path": str(config_path),
            "history_path": str(history_path)
        }
        
        logger.info("Model saved successfully", **file_paths)
        return file_paths
    
    async def save_model_metadata(
        self,
        model_name: str,
        version: str,
        file_paths: Dict[str, str],
        training_results: Dict,
        evaluation_results: Dict,
        description: str = ""
    ) -> MLModel:
        """
        Save model metadata to database.
        
        Args:
            model_name: Model name
            version: Model version
            file_paths: Paths to saved files
            training_results: Training results
            evaluation_results: Evaluation results
            description: Model description
            
        Returns:
            MLModel instance
        """
        async with AsyncSessionLocal() as session:
            # Create MLModel record
            ml_model = MLModel(
                name=model_name,
                version=version,
                model_type=self.model_type,
                status=ModelStatus.TRAINED,
                model_path=file_paths["model_path"],
                scaler_path=file_paths["scaler_path"],
                config_path=file_paths["config_path"],
                training_params=self.training_params,
                feature_columns=self.feature_engineer.feature_columns,
                target_column="close",
                train_accuracy=training_results.get("final_train_r2"),
                val_accuracy=training_results.get("final_val_r2"),
                test_accuracy=evaluation_results.get("test_r2"),
                mse=evaluation_results.get("test_mse"),
                mae=evaluation_results.get("test_mae"),
                training_started_at=datetime.now(),
                training_completed_at=datetime.now(),
                description=description
            )
            
            session.add(ml_model)
            await session.commit()
            await session.refresh(ml_model)
            
            logger.info("Model metadata saved to database",
                       model_id=ml_model.id,
                       model_name=model_name,
                       version=version)
            
            return ml_model
    
    def load_model(self, model_path: str, config_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to model file
            config_path: Path to config file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        self.model_type = config["model_type"]
        self.model_params = config["model_params"]
        self.feature_engineer.feature_columns = config["feature_columns"]
        
        # Create and load model
        self.create_model(input_size=len(config["feature_columns"]))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info("Model loaded successfully",
                   model_path=model_path,
                   model_type=self.model_type)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features array
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Transform features
        X_scaled = self.feature_engineer.transform_features(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
