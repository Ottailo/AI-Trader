"""Add ML models and backtest results

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ml_models table
    op.create_table('ml_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('model_path', sa.String(length=500), nullable=True),
        sa.Column('scaler_path', sa.String(length=500), nullable=True),
        sa.Column('config_path', sa.String(length=500), nullable=True),
        sa.Column('training_params', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('feature_columns', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('target_column', sa.String(length=100), nullable=True),
        sa.Column('train_accuracy', sa.Float(), nullable=True),
        sa.Column('val_accuracy', sa.Float(), nullable=True),
        sa.Column('test_accuracy', sa.Float(), nullable=True),
        sa.Column('mse', sa.Float(), nullable=True),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('training_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('training_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('training_duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_models_id'), 'ml_models', ['id'], unique=False)
    op.create_index(op.f('ix_ml_models_name'), 'ml_models', ['name'], unique=False)
    
    # Create backtest_results table
    op.create_table('backtest_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('strategy_name', sa.String(length=255), nullable=False),
        sa.Column('strategy_params', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbols', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('total_return', sa.Float(), nullable=True),
        sa.Column('annualized_return', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('volatility', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('profit_factor', sa.Float(), nullable=True),
        sa.Column('calmar_ratio', sa.Float(), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('winning_trades', sa.Integer(), nullable=True),
        sa.Column('losing_trades', sa.Integer(), nullable=True),
        sa.Column('avg_win', sa.Float(), nullable=True),
        sa.Column('avg_loss', sa.Float(), nullable=True),
        sa.Column('largest_win', sa.Float(), nullable=True),
        sa.Column('largest_loss', sa.Float(), nullable=True),
        sa.Column('var_95', sa.Float(), nullable=True),
        sa.Column('cvar_95', sa.Float(), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('alpha', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_time_seconds', sa.Float(), nullable=True),
        sa.Column('equity_curve', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('trade_log', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('monthly_returns', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_backtest_results_id'), 'backtest_results', ['id'], unique=False)
    op.create_index(op.f('ix_backtest_results_name'), 'backtest_results', ['name'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_backtest_results_name'), table_name='backtest_results')
    op.drop_index(op.f('ix_backtest_results_id'), table_name='backtest_results')
    op.drop_table('backtest_results')
    op.drop_index(op.f('ix_ml_models_name'), table_name='ml_models')
    op.drop_index(op.f('ix_ml_models_id'), table_name='ml_models')
    op.drop_table('ml_models')
