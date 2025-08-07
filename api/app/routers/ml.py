"""
ML model training and prediction endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import pandas as pd
import structlog

from app.ml.training import ModelTrainer
from app.ml.prediction import PredictionService
from app.backtesting.engine import BacktestingEngine, BacktestConfig
from app.backtesting.strategies import MLStrategy, MeanReversionStrategy, MomentumStrategy
from app.services.data_ingestion import DataIngestionService
from app.models import MLModel, BacktestResult, ModelStatus, BacktestStatus
from app.database import AsyncSessionLocal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Initialize services
prediction_service = PredictionService()
data_ingestion_service = DataIngestionService()


# Pydantic models
class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    name: str = Field(..., description="Model name")
    version: str = Field(default="1.0.0", description="Model version")
    model_type: str = Field(default="lstm", description="Model type (lstm, gru)")
    symbols: List[str] = Field(..., description="List of symbols to train on")
    start_date: Optional[str] = Field(None, description="Training start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Training end date (YYYY-MM-DD)")
    model_params: Optional[Dict[str, Any]] = Field(default={}, description="Model hyperparameters")
    training_params: Optional[Dict[str, Any]] = Field(default={}, description="Training parameters")
    description: Optional[str] = Field(None, description="Model description")


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    symbol: str = Field(..., description="Trading symbol")
    model_name: Optional[str] = Field(None, description="Specific model name")
    sequence_length: int = Field(default=60, description="Input sequence length")
    confidence_threshold: float = Field(default=0.02, description="Confidence threshold for signals")


class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    strategy_name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(..., description="Strategy type (ml, mean_reversion, momentum)")
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    model_name: Optional[str] = Field(None, description="ML model name (for ML strategy)")
    strategy_params: Optional[Dict[str, Any]] = Field(default={}, description="Strategy parameters")
    backtest_config: Optional[Dict[str, Any]] = Field(default={}, description="Backtest configuration")


class ModelResponse(BaseModel):
    """Response model for ML model information."""
    id: int
    name: str
    version: str
    model_type: str
    status: str
    description: Optional[str]
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    test_accuracy: Optional[float]
    created_at: datetime
    updated_at: Optional[datetime]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    symbol: str
    current_price: float
    predicted_price: Optional[float]
    signal: str
    confidence: float
    expected_return: float
    timestamp: datetime


class BacktestResponse(BaseModel):
    """Response model for backtest results."""
    id: int
    name: str
    status: str
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    total_return: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    total_trades: Optional[int]
    win_rate: Optional[float]
    created_at: datetime


# Dependency
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/train", response_model=Dict[str, Any])
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Train a new ML model.
    
    This endpoint starts model training in the background.
    """
    try:
        # Check if model already exists
        query = select(MLModel).where(
            MLModel.name == request.name,
            MLModel.version == request.version
        )
        result = await db.execute(query)
        existing_model = result.scalar_one_or_none()
        
        if existing_model:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.name} v{request.version} already exists"
            )
        
        # Create model record
        ml_model = MLModel(
            name=request.name,
            version=request.version,
            model_type=request.model_type,
            status=ModelStatus.TRAINING,
            training_params=request.training_params,
            description=request.description
        )
        
        db.add(ml_model)
        await db.commit()
        await db.refresh(ml_model)
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            ml_model.id,
            request
        )
        
        logger.info("Model training started",
                   model_id=ml_model.id,
                   model_name=request.name,
                   symbols=request.symbols)
        
        return {
            "message": "Model training started",
            "model_id": ml_model.id,
            "model_name": request.name,
            "status": "training"
        }
        
    except Exception as e:
        logger.error("Error starting model training", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _train_model_background(model_id: int, request: ModelTrainingRequest):
    """Background task for model training."""
    try:
        async with AsyncSessionLocal() as db:
            # Get model record
            query = select(MLModel).where(MLModel.id == model_id)
            result = await db.execute(query)
            ml_model = result.scalar_one_or_none()
            
            if not ml_model:
                logger.error("Model not found", model_id=model_id)
                return
            
            # Update status
            ml_model.status = ModelStatus.TRAINING
            ml_model.training_started_at = datetime.now()
            await db.commit()
            
            # Fetch historical data
            all_data = []
            for symbol in request.symbols:
                data = await data_ingestion_service.get_historical_data(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                if data is not None:
                    all_data.append(data)
            
            if not all_data:
                raise ValueError("No data available for training")
            
            # Combine data
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            # Initialize trainer
            trainer = ModelTrainer(
                model_type=request.model_type,
                model_params=request.model_params,
                training_params=request.training_params
            )
            
            # Create model
            trainer.create_model(input_size=len(trainer.feature_engineer.feature_columns))
            
            # Prepare data
            train_loader, val_loader, test_loader, feature_cols = trainer.prepare_data(
                combined_data,
                target_column="close",
                sequence_length=60
            )
            
            # Train model
            training_results = trainer.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=request.name,
                epochs=request.training_params.get("epochs", 100),
                learning_rate=request.training_params.get("learning_rate", 0.001)
            )
            
            # Evaluate model
            evaluation_results = trainer.evaluate_model(test_loader)
            
            # Save model
            file_paths = trainer.save_model(
                model_name=request.name,
                version=request.version,
                training_results=training_results,
                evaluation_results=evaluation_results
            )
            
            # Update model record
            ml_model.status = ModelStatus.TRAINED
            ml_model.training_completed_at = datetime.now()
            ml_model.training_duration_seconds = (
                ml_model.training_completed_at - ml_model.training_started_at
            ).total_seconds()
            ml_model.model_path = file_paths["model_path"]
            ml_model.scaler_path = file_paths["scaler_path"]
            ml_model.config_path = file_paths["config_path"]
            ml_model.feature_columns = feature_cols
            ml_model.target_column = "close"
            ml_model.train_accuracy = training_results.get("final_train_r2")
            ml_model.val_accuracy = training_results.get("final_val_r2")
            ml_model.test_accuracy = evaluation_results.get("test_r2")
            ml_model.mse = evaluation_results.get("test_mse")
            ml_model.mae = evaluation_results.get("test_mae")
            
            await db.commit()
            
            logger.info("Model training completed",
                       model_id=model_id,
                       model_name=request.name,
                       final_accuracy=evaluation_results.get("test_r2"))
            
    except Exception as e:
        logger.error("Error in background training", error=str(e))
        # Update model status to failed
        async with AsyncSessionLocal() as db:
            query = select(MLModel).where(MLModel.id == model_id)
            result = await db.execute(query)
            ml_model = result.scalar_one_or_none()
            if ml_model:
                ml_model.status = ModelStatus.FAILED
                await db.commit()


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Make a price prediction using the active model.
    """
    try:
        # Fetch historical data
        data = await data_ingestion_service.get_historical_data(
            symbol=request.symbol,
            limit=request.sequence_length + 100  # Extra data for feature calculation
        )
        
        if data is None or len(data) < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for symbol {request.symbol}"
            )
        
        # Make prediction
        prediction_result = await prediction_service.predict_signal(
            data=data,
            model_name=request.model_name,
            confidence_threshold=request.confidence_threshold,
            sequence_length=request.sequence_length
        )
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=prediction_result["current_price"],
            predicted_price=prediction_result["predicted_price"],
            signal=prediction_result["signal"],
            confidence=prediction_result["confidence"],
            expected_return=prediction_result["expected_return"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error("Error making prediction", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Run a backtest for a trading strategy.
    """
    try:
        # Create backtest record
        backtest = BacktestResult(
            name=request.strategy_name,
            strategy_name=request.strategy_type,
            strategy_params=request.strategy_params,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date),
            symbols=request.symbols,
            status=BacktestStatus.RUNNING
        )
        
        db.add(backtest)
        await db.commit()
        await db.refresh(backtest)
        
        # Start backtest in background
        background_tasks.add_task(
            _run_backtest_background,
            backtest.id,
            request
        )
        
        logger.info("Backtest started",
                   backtest_id=backtest.id,
                   strategy_name=request.strategy_name,
                   symbols=request.symbols)
        
        return BacktestResponse(
            id=backtest.id,
            name=backtest.name,
            status=backtest.status,
            strategy_name=backtest.strategy_name,
            symbols=backtest.symbols,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            total_return=backtest.total_return,
            sharpe_ratio=backtest.sharpe_ratio,
            max_drawdown=backtest.max_drawdown,
            total_trades=backtest.total_trades,
            win_rate=backtest.win_rate,
            created_at=backtest.created_at
        )
        
    except Exception as e:
        logger.error("Error starting backtest", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _run_backtest_background(backtest_id: int, request: BacktestRequest):
    """Background task for running backtest."""
    try:
        async with AsyncSessionLocal() as db:
            # Get backtest record
            query = select(BacktestResult).where(BacktestResult.id == backtest_id)
            result = await db.execute(query)
            backtest = result.scalar_one_or_none()
            
            if not backtest:
                logger.error("Backtest not found", backtest_id=backtest_id)
                return
            
            # Fetch historical data
            all_data = []
            for symbol in request.symbols:
                data = await data_ingestion_service.get_historical_data(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                if data is not None:
                    all_data.append(data)
            
            if not all_data:
                raise ValueError("No data available for backtest")
            
            # Combine data
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            # Create strategy
            if request.strategy_type == "ml":
                # Create ML strategy
                async def model_predictor(data):
                    return await prediction_service.predict_price(
                        data=data,
                        model_name=request.model_name
                    )
                
                strategy = MLStrategy(
                    name=request.strategy_name,
                    model_predictor=model_predictor,
                    params=request.strategy_params
                )
            elif request.strategy_type == "mean_reversion":
                strategy = MeanReversionStrategy(
                    name=request.strategy_name,
                    params=request.strategy_params
                )
            elif request.strategy_type == "momentum":
                strategy = MomentumStrategy(
                    name=request.strategy_name,
                    params=request.strategy_params
                )
            else:
                raise ValueError(f"Unsupported strategy type: {request.strategy_type}")
            
            # Create backtesting engine
            config = BacktestConfig(**request.backtest_config)
            engine = BacktestingEngine(config=config)
            
            # Run backtest
            results = engine.run_backtest(
                strategy=strategy,
                data=combined_data,
                symbols=request.symbols,
                start_date=datetime.fromisoformat(request.start_date),
                end_date=datetime.fromisoformat(request.end_date)
            )
            
            # Update backtest record
            backtest.status = BacktestStatus.COMPLETED
            backtest.completed_at = datetime.now()
            backtest.execution_time_seconds = (
                backtest.completed_at - backtest.started_at
            ).total_seconds()
            
            # Update performance metrics
            performance = results.get("performance_metrics", {})
            backtest.total_return = performance.get("total_return")
            backtest.annualized_return = performance.get("annualized_return")
            backtest.sharpe_ratio = performance.get("annualized_sharpe")
            backtest.max_drawdown = performance.get("max_drawdown")
            backtest.volatility = performance.get("annualized_volatility")
            backtest.win_rate = performance.get("win_rate")
            
            # Update trade statistics
            trade_stats = results.get("trade_statistics", {})
            backtest.total_trades = trade_stats.get("total_trades")
            backtest.winning_trades = trade_stats.get("winning_trades")
            backtest.losing_trades = trade_stats.get("losing_trades")
            backtest.avg_win = trade_stats.get("avg_win")
            backtest.avg_loss = trade_stats.get("avg_loss")
            
            # Store results
            backtest.equity_curve = results.get("equity_curve")
            backtest.trade_log = results.get("trades")
            
            await db.commit()
            
            logger.info("Backtest completed",
                       backtest_id=backtest_id,
                       total_return=backtest.total_return,
                       sharpe_ratio=backtest.sharpe_ratio)
            
    except Exception as e:
        logger.error("Error in background backtest", error=str(e))
        # Update backtest status to failed
        async with AsyncSessionLocal() as db:
            query = select(BacktestResult).where(BacktestResult.id == backtest_id)
            result = await db.execute(query)
            backtest = result.scalar_one_or_none()
            if backtest:
                backtest.status = BacktestStatus.FAILED
                await db.commit()


@router.get("/models", response_model=List[ModelResponse])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all ML models."""
    try:
        query = select(MLModel).order_by(MLModel.created_at.desc())
        result = await db.execute(query)
        models = result.scalars().all()
        
        return [
            ModelResponse(
                id=model.id,
                name=model.name,
                version=model.version,
                model_type=model.model_type,
                status=model.status,
                description=model.description,
                train_accuracy=model.train_accuracy,
                val_accuracy=model.val_accuracy,
                test_accuracy=model.test_accuracy,
                created_at=model.created_at,
                updated_at=model.updated_at
            )
            for model in models
        ]
        
    except Exception as e:
        logger.error("Error listing models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed information about a specific model."""
    try:
        query = select(MLModel).where(MLModel.id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get additional info from prediction service
        model_info = await prediction_service.get_model_info(model.name)
        
        return model_info
        
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/deploy")
async def deploy_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Deploy a trained model."""
    try:
        query = select(MLModel).where(MLModel.id == model_id)
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if model.status != ModelStatus.TRAINED:
            raise HTTPException(
                status_code=400,
                detail=f"Model must be trained before deployment. Current status: {model.status}"
            )
        
        # Deactivate other models of the same type
        await db.execute(
            select(MLModel).where(
                MLModel.model_type == model.model_type,
                MLModel.status == ModelStatus.DEPLOYED
            ).update({MLModel.status: ModelStatus.ARCHIVED})
        )
        
        # Deploy this model
        model.status = ModelStatus.DEPLOYED
        await db.commit()
        
        # Clear prediction service cache
        prediction_service.clear_cache()
        
        logger.info("Model deployed", model_id=model_id, model_name=model.name)
        
        return {"message": "Model deployed successfully", "model_id": model_id}
        
    except Exception as e:
        logger.error("Error deploying model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtests", response_model=List[BacktestResponse])
async def list_backtests(db: AsyncSession = Depends(get_db)):
    """List all backtest results."""
    try:
        query = select(BacktestResult).order_by(BacktestResult.created_at.desc())
        result = await db.execute(query)
        backtests = result.scalars().all()
        
        return [
            BacktestResponse(
                id=backtest.id,
                name=backtest.name,
                status=backtest.status,
                strategy_name=backtest.strategy_name,
                symbols=backtest.symbols,
                start_date=backtest.start_date,
                end_date=backtest.end_date,
                total_return=backtest.total_return,
                sharpe_ratio=backtest.sharpe_ratio,
                max_drawdown=backtest.max_drawdown,
                total_trades=backtest.total_trades,
                win_rate=backtest.win_rate,
                created_at=backtest.created_at
            )
            for backtest in backtests
        ]
        
    except Exception as e:
        logger.error("Error listing backtests", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtests/{backtest_id}", response_model=Dict[str, Any])
async def get_backtest_results(backtest_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed backtest results."""
    try:
        query = select(BacktestResult).where(BacktestResult.id == backtest_id)
        result = await db.execute(query)
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        return {
            "id": backtest.id,
            "name": backtest.name,
            "status": backtest.status,
            "strategy_name": backtest.strategy_name,
            "strategy_params": backtest.strategy_params,
            "symbols": backtest.symbols,
            "start_date": backtest.start_date,
            "end_date": backtest.end_date,
            "performance_metrics": {
                "total_return": backtest.total_return,
                "annualized_return": backtest.annualized_return,
                "sharpe_ratio": backtest.sharpe_ratio,
                "max_drawdown": backtest.max_drawdown,
                "volatility": backtest.volatility,
                "win_rate": backtest.win_rate,
                "profit_factor": backtest.profit_factor,
                "calmar_ratio": backtest.calmar_ratio
            },
            "trade_statistics": {
                "total_trades": backtest.total_trades,
                "winning_trades": backtest.winning_trades,
                "losing_trades": backtest.losing_trades,
                "avg_win": backtest.avg_win,
                "avg_loss": backtest.avg_loss,
                "largest_win": backtest.largest_win,
                "largest_loss": backtest.largest_loss
            },
            "risk_metrics": {
                "var_95": backtest.var_95,
                "cvar_95": backtest.cvar_95,
                "beta": backtest.beta,
                "alpha": backtest.alpha
            },
            "execution_info": {
                "started_at": backtest.started_at,
                "completed_at": backtest.completed_at,
                "execution_time_minutes": backtest.execution_time_minutes
            },
            "equity_curve": backtest.equity_curve,
            "trade_log": backtest.trade_log,
            "monthly_returns": backtest.monthly_returns
        }
        
    except Exception as e:
        logger.error("Error getting backtest results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
