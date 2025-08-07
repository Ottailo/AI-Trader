"""
Prediction service for ML models.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import structlog

from app.ml.models import LSTMModel, GRUModel
from app.ml.feature_engineering import FeatureEngineer
from app.models import MLModel, ModelStatus
from app.database import AsyncSessionLocal
from sqlalchemy import select

logger = structlog.get_logger()


class PredictionService:
    """
    Service for making predictions using trained ML models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the prediction service.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Cache for loaded models
        self.model_cache = {}
        self.feature_engineer_cache = {}
        
        logger.info("Prediction service initialized", models_dir=str(models_dir))
    
    async def get_active_model(self, model_name: Optional[str] = None) -> Optional[MLModel]:
        """
        Get the currently active model from database.
        
        Args:
            model_name: Specific model name (optional)
            
        Returns:
            Active MLModel instance or None
        """
        async with AsyncSessionLocal() as session:
            if model_name:
                query = select(MLModel).where(
                    MLModel.name == model_name,
                    MLModel.status == ModelStatus.DEPLOYED
                )
            else:
                query = select(MLModel).where(MLModel.status == ModelStatus.DEPLOYED)
            
            result = await session.execute(query)
            model = result.scalar_one_or_none()
            
            return model
    
    def load_model(self, model_path: str, config_path: str) -> Tuple[torch.nn.Module, FeatureEngineer]:
        """
        Load a trained model and its feature engineer.
        
        Args:
            model_path: Path to model file
            config_path: Path to config file
            
        Returns:
            Tuple of (model, feature_engineer)
        """
        # Check cache first
        cache_key = f"{model_path}_{config_path}"
        if cache_key in self.model_cache:
            logger.debug("Using cached model", cache_key=cache_key)
            return self.model_cache[cache_key]
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create feature engineer
        feature_engineer = FeatureEngineer(scaler_type=config.get("scaler_type", "standard"))
        feature_engineer.feature_columns = config["feature_columns"]
        
        # Load scaler
        scaler_path = config_path.replace("config.json", "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                feature_engineer.scaler = pickle.load(f)
                feature_engineer.is_fitted = True
        
        # Create and load model
        model_type = config["model_type"]
        model_params = config.get("model_params", {})
        
        if model_type == "lstm":
            model = LSTMModel(
                input_size=len(config["feature_columns"]),
                **model_params
            )
        elif model_type == "gru":
            model = GRUModel(
                input_size=len(config["feature_columns"]),
                **model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Cache the loaded model
        self.model_cache[cache_key] = (model, feature_engineer)
        
        logger.info("Model loaded successfully",
                   model_path=model_path,
                   model_type=model_type,
                   feature_count=len(config["feature_columns"]))
        
        return model, feature_engineer
    
    async def predict_price(
        self,
        data: pd.DataFrame,
        model_name: Optional[str] = None,
        sequence_length: int = 60
    ) -> Optional[float]:
        """
        Predict next price using the active model.
        
        Args:
            data: Historical market data
            model_name: Specific model name (optional)
            sequence_length: Length of input sequence
            
        Returns:
            Predicted price or None
        """
        try:
            # Get active model
            ml_model = await self.get_active_model(model_name)
            if not ml_model:
                logger.warning("No active model found")
                return None
            
            # Load model
            model, feature_engineer = self.load_model(
                ml_model.model_path,
                ml_model.config_path
            )
            
            # Prepare features
            df_features, feature_cols = feature_engineer.prepare_features(data)
            
            if len(df_features) < sequence_length:
                logger.warning("Insufficient data for prediction",
                              data_length=len(df_features),
                              required_length=sequence_length)
                return None
            
            # Create input sequence
            input_sequence = df_features[feature_cols].iloc[-sequence_length:].values
            
            # Transform features
            input_sequence_scaled = feature_engineer.transform_features(
                input_sequence.reshape(1, sequence_length, -1)
            )
            
            # Make prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_sequence_scaled)
                prediction = model(input_tensor)
                predicted_price = prediction.item()
            
            logger.info("Price prediction completed",
                       model_name=ml_model.name,
                       predicted_price=predicted_price,
                       current_price=data['close'].iloc[-1])
            
            return predicted_price
            
        except Exception as e:
            logger.error("Error making price prediction", error=str(e))
            return None
    
    async def predict_signal(
        self,
        data: pd.DataFrame,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.02,
        sequence_length: int = 60
    ) -> Dict[str, Any]:
        """
        Predict trading signal using the active model.
        
        Args:
            data: Historical market data
            model_name: Specific model name (optional)
            confidence_threshold: Minimum confidence for signal generation
            sequence_length: Length of input sequence
            
        Returns:
            Signal prediction dictionary
        """
        try:
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Make price prediction
            predicted_price = await self.predict_price(data, model_name, sequence_length)
            
            if predicted_price is None:
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "current_price": current_price,
                    "predicted_price": None,
                    "expected_return": 0.0,
                    "reason": "insufficient_data"
                }
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Generate signal based on confidence threshold
            if abs(expected_return) < confidence_threshold:
                signal = "hold"
                confidence = 0.0
            elif expected_return > 0:
                signal = "buy"
                confidence = min(abs(expected_return) / confidence_threshold, 1.0)
            else:
                signal = "sell"
                confidence = min(abs(expected_return) / confidence_threshold, 1.0)
            
            result = {
                "signal": signal,
                "confidence": confidence,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "reason": "model_prediction"
            }
            
            logger.info("Signal prediction completed",
                       signal=signal,
                       confidence=confidence,
                       expected_return=expected_return)
            
            return result
            
        except Exception as e:
            logger.error("Error making signal prediction", error=str(e))
            return {
                "signal": "hold",
                "confidence": 0.0,
                "current_price": data['close'].iloc[-1] if len(data) > 0 else 0,
                "predicted_price": None,
                "expected_return": 0.0,
                "reason": "error"
            }
    
    async def predict_multiple_steps(
        self,
        data: pd.DataFrame,
        steps: int = 5,
        model_name: Optional[str] = None,
        sequence_length: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple future price steps.
        
        Args:
            data: Historical market data
            steps: Number of steps to predict
            model_name: Specific model name (optional)
            sequence_length: Length of input sequence
            
        Returns:
            List of predictions
        """
        try:
            # Get active model
            ml_model = await self.get_active_model(model_name)
            if not ml_model:
                logger.warning("No active model found")
                return []
            
            # Load model
            model, feature_engineer = self.load_model(
                ml_model.model_path,
                ml_model.config_path
            )
            
            # Prepare features
            df_features, feature_cols = feature_engineer.prepare_features(data)
            
            if len(df_features) < sequence_length:
                logger.warning("Insufficient data for multi-step prediction")
                return []
            
            predictions = []
            current_data = df_features.copy()
            
            for step in range(steps):
                # Create input sequence
                input_sequence = current_data[feature_cols].iloc[-sequence_length:].values
                
                # Transform features
                input_sequence_scaled = feature_engineer.transform_features(
                    input_sequence.reshape(1, sequence_length, -1)
                )
                
                # Make prediction
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_sequence_scaled)
                    prediction = model(input_tensor)
                    predicted_price = prediction.item()
                
                # Add prediction to results
                predictions.append({
                    "step": step + 1,
                    "predicted_price": predicted_price,
                    "timestamp": data.index[-1] + pd.Timedelta(days=step + 1)
                })
                
                # Update data for next prediction (simplified approach)
                # In practice, you might want to use a more sophisticated method
                new_row = current_data.iloc[-1].copy()
                new_row['close'] = predicted_price
                # Update other OHLCV values (simplified)
                new_row['open'] = predicted_price
                new_row['high'] = predicted_price
                new_row['low'] = predicted_price
                new_row['volume'] = current_data['volume'].iloc[-1]  # Keep same volume
                
                # Add new row to data
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                
                # Recalculate features for the new data point
                # This is a simplified approach - in practice, you'd want to recalculate all features
                current_data = feature_engineer.calculate_technical_indicators(current_data)
            
            logger.info("Multi-step prediction completed",
                       steps=steps,
                       predictions_count=len(predictions))
            
            return predictions
            
        except Exception as e:
            logger.error("Error making multi-step prediction", error=str(e))
            return []
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the active model.
        
        Args:
            model_name: Specific model name (optional)
            
        Returns:
            Model information dictionary
        """
        try:
            # Get active model
            ml_model = await self.get_active_model(model_name)
            if not ml_model:
                return {"error": "No active model found"}
            
            # Load model configuration
            with open(ml_model.config_path, 'r') as f:
                config = json.load(f)
            
            info = {
                "model_id": ml_model.id,
                "name": ml_model.name,
                "version": ml_model.version,
                "model_type": ml_model.model_type,
                "status": ml_model.status,
                "description": ml_model.description,
                "training_params": ml_model.training_params,
                "feature_columns": ml_model.feature_columns,
                "target_column": ml_model.target_column,
                "performance_metrics": {
                    "train_accuracy": ml_model.train_accuracy,
                    "val_accuracy": ml_model.val_accuracy,
                    "test_accuracy": ml_model.test_accuracy,
                    "mse": ml_model.mse,
                    "mae": ml_model.mae
                },
                "training_info": {
                    "training_started_at": ml_model.training_started_at,
                    "training_completed_at": ml_model.training_completed_at,
                    "training_duration_minutes": ml_model.training_duration_minutes
                },
                "model_config": config
            }
            
            return info
            
        except Exception as e:
            logger.error("Error getting model info", error=str(e))
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.model_cache.clear()
        self.feature_engineer_cache.clear()
        logger.info("Model cache cleared")
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            async with AsyncSessionLocal() as session:
                query = select(MLModel).order_by(MLModel.created_at.desc())
                result = await session.execute(query)
                models = result.scalars().all()
                
                model_list = []
                for model in models:
                    model_info = {
                        "id": model.id,
                        "name": model.name,
                        "version": model.version,
                        "model_type": model.model_type,
                        "status": model.status,
                        "description": model.description,
                        "created_at": model.created_at,
                        "updated_at": model.updated_at,
                        "performance": {
                            "train_accuracy": model.train_accuracy,
                            "val_accuracy": model.val_accuracy,
                            "test_accuracy": model.test_accuracy
                        }
                    }
                    model_list.append(model_info)
                
                return model_list
                
        except Exception as e:
            logger.error("Error listing models", error=str(e))
            return []
