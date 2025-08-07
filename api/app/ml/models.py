"""
Neural network models for time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        output_size: Number of output features (usually 1 for price prediction)
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from the sequence
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers with activation and dropout
        out = F.relu(self.fc1(lstm_out))
        out = self.bn1(out)
        out = self.dropout_layer(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout_layer(out)
        
        # Final output layer
        out = self.fc3(out)
        
        return out
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "model_type": "lstm",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional
        }


class GRUModel(nn.Module):
    """
    GRU model for time series prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of GRU hidden units
        num_layers: Number of GRU layers
        output_size: Number of output features (usually 1 for price prediction)
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional GRU
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size based on bidirectional setting
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the last output from the sequence
        gru_out = gru_out[:, -1, :]
        
        # Fully connected layers with activation and dropout
        out = F.relu(self.fc1(gru_out))
        out = self.bn1(out)
        out = self.dropout_layer(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout_layer(out)
        
        # Final output layer
        out = self.fc3(out)
        
        return out
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "model_type": "gru",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional
        }


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence models.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            Weighted output tensor of shape (batch_size, hidden_size)
        """
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        
        # Apply attention weights
        weighted_output = torch.sum(attention_weights * x, dim=1)
        
        return weighted_output


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    
    Args:
        input_size: Number of input features
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        output_size: Number of output features
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(1000, d_model)  # Max sequence length of 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoding = self.pos_encoder(positions)
        x = x + pos_encoding
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "model_type": "transformer",
            "input_size": self.input_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout
        }
