import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Embedding, Concatenate, BatchNormalization, Add, Reshape
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from typing import Dict, Tuple, Optional
import numpy as np
from loguru import logger

class UniversalModelArchitectures:
    """
    Universal model architectures with symbol embeddings for multi-symbol training.
    Supports LSTM, CNN, and Transformer models with shared representations.
    """
    
    def __init__(self, num_symbols: int, symbol_embedding_dim: int = 32):
        """
        Initialize universal model architectures.
        
        Args:
            num_symbols: Number of unique symbols in the trading universe
            symbol_embedding_dim: Dimension of symbol embeddings
        """
        self.num_symbols = num_symbols
        self.symbol_embedding_dim = symbol_embedding_dim
        logger.info(f"Initialized universal architectures for {num_symbols} symbols with {symbol_embedding_dim}D embeddings")
    
    def create_universal_lstm(
        self,
        sequence_length: int,
        feature_dim: int,
        config: Dict
    ) -> Model:
        """
        Create universal LSTM model with symbol embeddings.
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features per timestep
            config: Model configuration parameters
            
        Returns:
            Compiled universal LSTM model
        """
        # Extract parameters
        units = config.get('units', 50)
        dropout = config.get('dropout', 0.2)
        l2_reg = config.get('l2_reg', 0.01)
        learning_rate = config.get('learning_rate', 0.001)
        
        # Input layers
        feature_input = Input(shape=(sequence_length, feature_dim), name='feature_input')
        symbol_input = Input(shape=(), dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # LSTM layers for feature processing
        lstm1 = LSTM(
            units=units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=dropout,
            kernel_regularizer=l2(l2_reg),
            name='lstm_1'
        )(feature_input)
        
        lstm2 = LSTM(
            units=units//2,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            kernel_regularizer=l2(l2_reg),
            name='lstm_2'
        )(lstm1)
        
        # Combine LSTM output with symbol embedding
        combined = Concatenate(name='feature_symbol_concat')([lstm2, symbol_embedding])
        
        # Dense layers for final processing
        dense1 = Dense(
            units=64,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_1'
        )(combined)
        dropout1 = Dropout(dropout, name='dropout_1')(dense1)
        
        dense2 = Dense(
            units=32,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_2'
        )(dropout1)
        dropout2 = Dropout(dropout, name='dropout_2')(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Create and compile model
        model = Model(
            inputs=[feature_input, symbol_input],
            outputs=output,
            name='universal_lstm'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created universal LSTM model with {model.count_params()} parameters")
        return model
    
    def create_universal_cnn(
        self,
        sequence_length: int,
        feature_dim: int,
        config: Dict
    ) -> Model:
        """
        Create universal CNN model with symbol embeddings.
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features per timestep
            config: Model configuration parameters
            
        Returns:
            Compiled universal CNN model
        """
        # Extract parameters
        filters = config.get('filters', 64)
        kernel_size = config.get('kernel_size', (3, 3))
        dropout = config.get('dropout', 0.3)
        l2_reg = config.get('l2_reg', 0.01)
        learning_rate = config.get('learning_rate', 0.0005)
        
        # Input layers
        feature_input = Input(shape=(sequence_length, feature_dim), name='feature_input')
        symbol_input = Input(shape=(), dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Reshape for CNN processing (add channel dimension)
        reshaped_input = Reshape((sequence_length, feature_dim, 1), name='reshape_for_cnn')(feature_input)
        
        # CNN layers for feature processing
        conv1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='conv2d_1'
        )(reshaped_input)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(conv1)
        
        conv2 = Conv2D(
            filters=filters*2,
            kernel_size=kernel_size,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='conv2d_2'
        )(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(conv2)
        
        # Flatten CNN output
        flattened = Flatten(name='flatten')(pool2)
        
        # Combine CNN output with symbol embedding
        combined = Concatenate(name='feature_symbol_concat')([flattened, symbol_embedding])
        
        # Dense layers for final processing
        dense1 = Dense(
            units=128,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_1'
        )(combined)
        dropout1 = Dropout(dropout, name='dropout_1')(dense1)
        
        dense2 = Dense(
            units=50,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_2'
        )(dropout1)
        dropout2 = Dropout(dropout, name='dropout_2')(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Create and compile model
        model = Model(
            inputs=[feature_input, symbol_input],
            outputs=output,
            name='universal_cnn'
        )
        
        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created universal CNN model with {model.count_params()} parameters")
        return model
    
    def create_universal_transformer(
        self,
        sequence_length: int,
        feature_dim: int,
        config: Dict
    ) -> Model:
        """
        Create universal Transformer model with symbol embeddings.
        
        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features per timestep
            config: Model configuration parameters
            
        Returns:
            Compiled universal Transformer model
        """
        # Extract parameters
        num_heads = config.get('num_heads', 2)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        learning_rate = config.get('learning_rate', 0.001)
        
        # Input layers
        feature_input = Input(shape=(sequence_length, feature_dim), name='feature_input')
        symbol_input = Input(shape=(), dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Transformer layers for feature processing
        x = feature_input
        
        for i in range(num_layers):
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=feature_dim // num_heads,
                name=f'multi_head_attention_{i+1}'
            )(x, x)
            
            # Add & Norm
            x = Add(name=f'add_{i+1}')([x, attention])
            x = LayerNormalization(name=f'layer_norm_{i+1}')(x)
            
            # Dropout
            x = Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D(name='global_average_pooling')(x)
        
        # Combine transformer output with symbol embedding
        combined = Concatenate(name='feature_symbol_concat')([pooled, symbol_embedding])
        
        # Dense layers for final processing
        dense1 = Dense(
            units=64,
            activation='relu',
            name='dense_1'
        )(combined)
        dropout1 = Dropout(dropout, name='final_dropout_1')(dense1)
        
        dense2 = Dense(
            units=32,
            activation='relu',
            name='dense_2'
        )(dropout1)
        dropout2 = Dropout(dropout, name='final_dropout_2')(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Create and compile model
        model = Model(
            inputs=[feature_input, symbol_input],
            outputs=output,
            name='universal_transformer'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created universal Transformer model with {model.count_params()} parameters")
        return model
    
    def create_symbol_specific_head(
        self,
        base_model: Model,
        symbol_id: int,
        config: Dict
    ) -> Model:
        """
        Create symbol-specific fine-tuning head for a universal base model.
        
        Args:
            base_model: Pre-trained universal base model
            symbol_id: ID of the specific symbol
            config: Fine-tuning configuration
            
        Returns:
            Model with symbol-specific head
        """
        # Freeze base model layers (except last few)
        layers_to_unfreeze = config.get('layers_to_unfreeze', 3)
        
        for i, layer in enumerate(base_model.layers):
            if i < len(base_model.layers) - layers_to_unfreeze:
                layer.trainable = False
            else:
                layer.trainable = True
        
        # Get the feature representation before the final dense layers
        feature_representation = base_model.layers[-4].output  # Before last 3 dense layers
        
        # Symbol-specific dense layers
        symbol_dense1 = Dense(
            units=32,
            activation='relu',
            name=f'symbol_{symbol_id}_dense_1'
        )(feature_representation)
        
        symbol_dropout = Dropout(
            config.get('dropout', 0.2),
            name=f'symbol_{symbol_id}_dropout'
        )(symbol_dense1)
        
        symbol_output = Dense(
            1,
            activation='sigmoid',
            name=f'symbol_{symbol_id}_output'
        )(symbol_dropout)
        
        # Create symbol-specific model
        symbol_model = Model(
            inputs=base_model.inputs,
            outputs=symbol_output,
            name=f'symbol_{symbol_id}_model'
        )
        
        # Compile with lower learning rate for fine-tuning
        symbol_model.compile(
            optimizer=Adam(learning_rate=config.get('fine_tune_lr', 0.0001), clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created symbol-specific model for symbol {symbol_id} with {symbol_model.count_params()} parameters")
        return symbol_model
    
    def get_model_summary(self, model: Model) -> Dict:
        """
        Get comprehensive model summary including architecture details.
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with model summary information
        """
        return {
            'name': model.name,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            'input_shapes': [input_layer.shape for input_layer in model.inputs],
            'output_shape': model.output.shape,
            'num_layers': len(model.layers),
            'optimizer': model.optimizer.__class__.__name__,
            'loss': model.loss,
            'metrics': model.metrics_names
        }