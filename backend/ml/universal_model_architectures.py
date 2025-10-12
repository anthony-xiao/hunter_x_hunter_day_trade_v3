import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Embedding, Concatenate, BatchNormalization, Add, Reshape
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from typing import Dict, Tuple, Optional, Any
import numpy as np
from loguru import logger

# Statistical model imports
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
from pathlib import Path

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
        
        # Handle both list and single integer units configuration
        if isinstance(units, list):
            units_list = units
        else:
            # Backward compatibility: convert single integer to list
            units_list = [units, units//2]
        
        # Input layers
        feature_input = Input(shape=(sequence_length, feature_dim), name='feature_input')
        symbol_input = Input(shape=(), dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # LSTM layers for feature processing - dynamically create based on units_list
        x = feature_input
        for i, layer_units in enumerate(units_list):
            return_sequences = (i < len(units_list) - 1)  # Return sequences for all but last layer
            x = LSTM(
                units=layer_units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=dropout,
                kernel_regularizer=l2(l2_reg),
                name=f'lstm_{i+1}'
            )(x)
        
        # x now contains the output from the final LSTM layer
        lstm_output = x
        
        # Combine LSTM output with symbol embedding
        combined = Concatenate(name='feature_symbol_concat')([lstm_output, symbol_embedding])
        
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
        
        # Log detailed model architecture
        logger.info(f"Created universal LSTM model with {model.count_params()} parameters")
        logger.info("LSTM Model Layer Structure:")
        for i, layer in enumerate(model.layers):
            logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__}) - Output Shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")
        
        # Validate expected layers exist
        expected_layers = ['feature_input', 'symbol_input', 'symbol_embedding', 'lstm_1', 'lstm_2', 'feature_symbol_concat']
        found_layers = [layer.name for layer in model.layers]
        for expected in expected_layers:
            if expected in found_layers:
                logger.info(f"✓ Expected layer '{expected}' found")
            else:
                logger.warning(f"✗ Expected layer '{expected}' NOT found")
        
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
        
        # Handle both list and single integer filters configuration
        if isinstance(filters, list):
            filters_list = filters
        else:
            # Backward compatibility: convert single integer to list
            filters_list = [filters, filters*2]
        
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
        
        # CNN layers for feature processing - dynamically create based on filters_list
        x = reshaped_input
        for i, layer_filters in enumerate(filters_list):
            # Adjust kernel size based on layer depth
            if i == 0:
                kernel = (3, 2)  # First layer kernel
                pool_size = (2, 1)
            else:
                kernel = (2, 2)  # Subsequent layers kernel
                pool_size = (2, 1)
            
            x = Conv2D(
                filters=layer_filters,
                kernel_size=kernel,
                activation='relu',
                kernel_regularizer=l2(l2_reg),
                name=f'conv2d_{i+1}'
            )(x)
            x = MaxPooling2D(pool_size=pool_size, name=f'max_pooling2d_{i+1}')(x)
        
        # x now contains the output from the final CNN layer
        cnn_output = x
        
        # Flatten CNN output
        flattened = Flatten(name='flatten')(cnn_output)
        
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
        
        # Log detailed model architecture
        logger.info(f"Created universal CNN model with {model.count_params()} parameters")
        logger.info("CNN Model Layer Structure:")
        for i, layer in enumerate(model.layers):
            logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__}) - Output Shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")
        
        # Validate expected layers exist
        expected_layers = ['feature_input', 'symbol_input', 'symbol_embedding', 'reshape_for_cnn', 'conv2d_1', 'conv2d_2', 'flatten', 'feature_symbol_concat']
        found_layers = [layer.name for layer in model.layers]
        for expected in expected_layers:
            if expected in found_layers:
                logger.info(f"✓ Expected layer '{expected}' found")
            else:
                logger.warning(f"✗ Expected layer '{expected}' NOT found")
        
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
        
        # Log detailed model architecture
        logger.info(f"Created universal Transformer model with {model.count_params()} parameters")
        logger.info("Transformer Model Layer Structure:")
        for i, layer in enumerate(model.layers):
            logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__}) - Output Shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")
        
        # Validate expected layers exist
        expected_layers = ['feature_input', 'symbol_input', 'symbol_embedding', 'global_average_pooling', 'feature_symbol_concat']
        found_layers = [layer.name for layer in model.layers]
        for expected in expected_layers:
            if expected in found_layers:
                logger.info(f"✓ Expected layer '{expected}' found")
            else:
                logger.warning(f"✗ Expected layer '{expected}' NOT found")
        
        return model
    
    def create_universal_dense(
        self,
        feature_dim: int,
        config: Dict,
        model_name: str = "universal_dense"
    ) -> Model:
        """
        Create universal dense neural network for 2D aggregated features.
        
        Args:
            feature_dim: Number of aggregated features
            config: Model configuration parameters
            model_name: Name for the model
            
        Returns:
            Compiled universal dense model
        """
        # Extract parameters with model-specific defaults
        dropout = config.get('dropout', 0.3)
        l2_reg = config.get('l2_reg', 0.01)
        learning_rate = config.get('learning_rate', 0.001)
        
        # Get layer configuration based on model type
        if 'lstm' in model_name.lower():
            # LSTM-inspired dense architecture
            units = config.get('units', [128, 64, 32])
            if isinstance(units, int):
                units = [units, units//2, units//4]
        elif 'cnn' in model_name.lower():
            # CNN-inspired dense architecture
            filters = config.get('filters', [64, 128])
            if isinstance(filters, int):
                filters = [filters, filters*2]
            units = [f*2 for f in filters] + [64, 32]  # Convert filters to dense units
        else:
            # Transformer or generic dense architecture
            units = config.get('units', [256, 128, 64])
            if isinstance(units, int):
                units = [units, units//2, units//4]
        
        # Input layers
        feature_input = Input(shape=(feature_dim,), name='feature_input')
        symbol_input = Input(shape=(), dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Dense layers for feature processing
        x = feature_input
        for i, layer_units in enumerate(units):
            x = Dense(
                units=layer_units,
                activation='relu',
                kernel_regularizer=l2(l2_reg),
                name=f'dense_feature_{i+1}'
            )(x)
            x = BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = Dropout(dropout, name=f'dropout_feature_{i+1}')(x)
        
        # Combine dense output with symbol embedding
        combined = Concatenate(name='feature_symbol_concat')([x, symbol_embedding])
        
        # Final dense layers
        dense1 = Dense(
            units=64,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_final_1'
        )(combined)
        dropout1 = Dropout(dropout, name='dropout_final_1')(dense1)
        
        dense2 = Dense(
            units=32,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='dense_final_2'
        )(dropout1)
        dropout2 = Dropout(dropout, name='dropout_final_2')(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Create and compile model
        model = Model(
            inputs=[feature_input, symbol_input],
            outputs=output,
            name=model_name
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Log detailed model architecture
        logger.info(f"Created {model_name} model with {model.count_params()} parameters")
        logger.info(f"Dense Model Layer Structure for {feature_dim} aggregated features:")
        for i, layer in enumerate(model.layers):
            logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__}) - Output Shape: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}")
        
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
        # === ARCHITECTURE VALIDATION LOGGING - BEFORE ===
        logger.info(f"\n=== SYMBOL-SPECIFIC HEAD CREATION START - Symbol {symbol_id} ===")
        logger.info(f"Base model architecture validation:")
        logger.info(f"  Name: {base_model.name}")
        logger.info(f"  Total params: {base_model.count_params()}")
        logger.info(f"  Input shapes: {[inp.shape for inp in base_model.inputs]}")
        logger.info(f"  Output shape: {base_model.output.shape}")
        logger.info(f"  Number of layers: {len(base_model.layers)}")
        
        # Validate base model inputs/outputs connectivity
        try:
            # Test base model connectivity by creating a dummy prediction
            import numpy as np
            dummy_inputs = []
            for inp in base_model.inputs:
                dummy_shape = [1] + list(inp.shape[1:])  # Add batch dimension
                dummy_inputs.append(np.random.random(dummy_shape))
            
            dummy_output = base_model.predict(dummy_inputs, verbose=0)
            logger.info(f"✓ Base model inputs/outputs connectivity validated - Output shape: {dummy_output.shape}")
        except Exception as e:
            logger.error(f"✗ Base model inputs/outputs connectivity FAILED: {e}")
            logger.error(f"  This indicates the base model itself may be corrupted!")
        
        # Extract architecture type from model name
        model_name = base_model.name.lower()
        if 'lstm' in model_name:
            architecture = 'lstm'
        elif 'cnn' in model_name:
            architecture = 'cnn'
        elif 'transformer' in model_name:
            architecture = 'transformer'
        else:
            logger.error(f"Cannot determine architecture from model name: {base_model.name}")
            raise ValueError(f"Unknown architecture in model name: {base_model.name}")
        
        logger.info(f"Detected architecture: {architecture} from model name: {base_model.name}")
        
        # Rebuild the model from scratch with symbol-specific head
        # This avoids connectivity issues from cloning and modifying
        logger.info(f"Rebuilding {architecture} model with symbol-specific head for symbol {symbol_id}...")
        try:
            if architecture == 'lstm':
                symbol_model = self._create_lstm_with_symbol_head(base_model, symbol_id, config)
            elif architecture == 'cnn':
                symbol_model = self._create_cnn_with_symbol_head(base_model, symbol_id, config)
            elif architecture == 'transformer':
                symbol_model = self._create_transformer_with_symbol_head(base_model, symbol_id, config)
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
                
            logger.info(f"✓ Successfully created symbol-specific model structure for symbol {symbol_id}")
            
        except Exception as e:
            logger.error(f"✗ Symbol-specific model creation FAILED for symbol {symbol_id}: {e}")
            raise
        
        # === ARCHITECTURE VALIDATION LOGGING - AFTER ===
        logger.info(f"\n=== SYMBOL-SPECIFIC MODEL VALIDATION - Symbol {symbol_id} ===")
        logger.info(f"Final model architecture:")
        logger.info(f"  Name: {symbol_model.name}")
        logger.info(f"  Total params: {symbol_model.count_params()}")
        logger.info(f"  Input shapes: {[inp.shape for inp in symbol_model.inputs]}")
        logger.info(f"  Output shape: {symbol_model.output.shape}")
        logger.info(f"  Number of layers: {len(symbol_model.layers)}")
        
        # Validate symbol-specific model inputs/outputs connectivity BEFORE compilation
        try:
            # Test symbol model connectivity by creating a dummy prediction
            import numpy as np
            dummy_inputs = []
            for inp in symbol_model.inputs:
                dummy_shape = [1] + list(inp.shape[1:])  # Add batch dimension
                dummy_inputs.append(np.random.random(dummy_shape))
            
            # Test model structure without compilation
            dummy_output = symbol_model(dummy_inputs, training=False)
            logger.info(f"✓ Symbol model inputs/outputs connectivity validated BEFORE compilation - Output shape: {dummy_output.shape}")
            
        except Exception as e:
            logger.error(f"✗ Symbol model inputs/outputs connectivity FAILED BEFORE compilation for symbol {symbol_id}: {e}")
            logger.error(f"  This indicates architectural corruption during head creation!")
            logger.error(f"  Model inputs: {[inp.name for inp in symbol_model.inputs]}")
            logger.error(f"  Model output: {symbol_model.output}")
            # Don't raise here, let's see if compilation fixes it
        
        # Compile with lower learning rate for fine-tuning
        logger.info(f"Compiling symbol-specific model for symbol {symbol_id}...")
        try:
            symbol_model.compile(
                optimizer=Adam(learning_rate=config.get('fine_tune_lr', 0.0001), clipnorm=1.0),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logger.info(f"✓ Successfully compiled symbol-specific model for symbol {symbol_id}")
            
        except Exception as e:
            logger.error(f"✗ Symbol model compilation FAILED for symbol {symbol_id}: {e}")
            raise
        
        # Final connectivity test AFTER compilation
        try:
            dummy_output = symbol_model.predict(dummy_inputs, verbose=0)
            logger.info(f"✓ Symbol model inputs/outputs connectivity validated AFTER compilation - Output shape: {dummy_output.shape}")
            
        except Exception as e:
            logger.error(f"✗ Symbol model inputs/outputs connectivity FAILED AFTER compilation for symbol {symbol_id}: {e}")
            logger.error(f"  This model will likely be corrupted when saved!")
        
        logger.info(f"=== SYMBOL-SPECIFIC HEAD CREATION COMPLETE - Symbol {symbol_id} ===\n")
        logger.info(f"Created symbol-specific model for symbol {symbol_id} with {symbol_model.count_params()} parameters")
        return symbol_model
    
    def _create_lstm_with_symbol_head(self, base_model, symbol_id, config):
        """Create LSTM model with symbol-specific head from scratch"""
        import logging
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate
        from tensorflow.keras.models import Model
        
        # Get input shapes from base model
        logger = logging.getLogger(__name__)
        try:
            # Base model has two inputs: [feature_input, symbol_input]
            feature_shape = base_model.inputs[0].shape[1:]  # Remove batch dimension
            symbol_shape = base_model.inputs[1].shape[1:]   # Remove batch dimension
            
            # Convert TensorShape to tuple if needed
            if hasattr(feature_shape, 'as_list'):
                feature_shape = tuple(feature_shape.as_list())
            elif not isinstance(feature_shape, tuple):
                feature_shape = tuple(feature_shape)
                
            if hasattr(symbol_shape, 'as_list'):
                symbol_shape = tuple(symbol_shape.as_list())
            elif not isinstance(symbol_shape, tuple):
                symbol_shape = tuple(symbol_shape)
                
            logger.info(f"Using feature_shape: {feature_shape}, symbol_shape: {symbol_shape}")
            
        except Exception as e:
            logger.error(f"Error getting input shapes: {e}")
            # Default fallback for LSTM
            feature_shape = (60, 5)  # sequence_length, features
            symbol_shape = ()        # symbol input shape
            logger.info(f"Using default LSTM shapes - feature: {feature_shape}, symbol: {symbol_shape}")
        
        # Create fresh input layers (matching base model)
        feature_input = Input(shape=feature_shape, name='feature_input')
        symbol_input = Input(shape=symbol_shape, dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding (matching base model)
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Recreate LSTM layers with same architecture as base model
        x = LSTM(50, return_sequences=True, name='lstm_1')(feature_input)
        lstm_features = LSTM(25, return_sequences=False, name='lstm_2')(x)
        
        # Combine LSTM output with symbol embedding (matching base model)
        combined = Concatenate(name='feature_symbol_concat')([lstm_features, symbol_embedding])
        
        # Add symbol-specific head
        symbol_dense1 = Dense(
            units=32,
            activation='relu',
            name=f'symbol_{symbol_id}_dense_1'
        )(combined)
        
        symbol_dropout = Dropout(
            config.get('dropout', 0.2),
            name=f'symbol_{symbol_id}_dropout'
        )(symbol_dense1)
        
        symbol_output = Dense(
            1,
            activation='sigmoid',
            name=f'symbol_{symbol_id}_output'
        )(symbol_dropout)
        
        # Create the model
        symbol_model = Model(
            inputs=[feature_input, symbol_input],
            outputs=symbol_output,
            name=f'symbol_{symbol_id}_lstm_model'
        )
        
        # Copy weights from base model (up to the feature extraction part)
        self._copy_base_weights(base_model, symbol_model, 'lstm')
        
        return symbol_model
    
    def _create_cnn_with_symbol_head(self, base_model, symbol_id, config):
        """Create CNN model with symbol-specific head from scratch"""
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Embedding, Concatenate, Reshape
        from tensorflow.keras.models import Model
        
        # Get input shapes from base model
        try:
            # Base model has two inputs: [feature_input, symbol_input]
            feature_shape = base_model.inputs[0].shape[1:]  # Remove batch dimension
            symbol_shape = base_model.inputs[1].shape[1:]   # Remove batch dimension
            
            # Convert TensorShape to tuple if needed
            if hasattr(feature_shape, 'as_list'):
                feature_shape = tuple(feature_shape.as_list())
            elif not isinstance(feature_shape, tuple):
                feature_shape = tuple(feature_shape)
                
            if hasattr(symbol_shape, 'as_list'):
                symbol_shape = tuple(symbol_shape.as_list())
            elif not isinstance(symbol_shape, tuple):
                symbol_shape = tuple(symbol_shape)
                
        except Exception as e:
            # Default fallback for CNN
            feature_shape = (60, 5)  # sequence_length, features
            symbol_shape = ()        # symbol input shape
        
        # Create fresh input layers (matching base model)
        feature_input = Input(shape=feature_shape, name='feature_input')
        symbol_input = Input(shape=symbol_shape, dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding (matching base model)
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Reshape for CNN processing (add channel dimension)
        reshaped_input = Reshape((feature_shape[0], feature_shape[1], 1), name='reshape_for_cnn')(feature_input)
        
        # Recreate CNN layers with same architecture as base model
        conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 2),  # Match base model kernel size
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='conv2d_1'
        )(reshaped_input)
        pool1 = MaxPooling2D(pool_size=(2, 1), name='max_pooling2d_1')(conv1)  # Match base model pooling
        
        conv2 = Conv2D(
            filters=128,
            kernel_size=(2, 2),  # Match base model kernel size
            activation='relu',
            kernel_regularizer=l2(0.01),
            name='conv2d_2'
        )(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 1), name='max_pooling2d_2')(conv2)  # Match base model pooling
        
        # Flatten CNN output
        flattened = Flatten(name='flatten')(pool2)
        
        # Combine CNN output with symbol embedding (matching base model)
        combined = Concatenate(name='feature_symbol_concat')([flattened, symbol_embedding])
        
        # Add symbol-specific head
        symbol_dense1 = Dense(
            units=32,
            activation='relu',
            name=f'symbol_{symbol_id}_dense_1'
        )(combined)
        
        symbol_dropout = Dropout(
            config.get('dropout', 0.2),
            name=f'symbol_{symbol_id}_dropout'
        )(symbol_dense1)
        
        symbol_output = Dense(
            1,
            activation='sigmoid',
            name=f'symbol_{symbol_id}_output'
        )(symbol_dropout)
        
        # Create the model
        symbol_model = Model(
            inputs=[feature_input, symbol_input],
            outputs=symbol_output,
            name=f'symbol_{symbol_id}_cnn_model'
        )
        
        # Copy weights from base model (up to the feature extraction part)
        self._copy_base_weights(base_model, symbol_model, 'cnn')
        
        return symbol_model
    
    def _create_transformer_with_symbol_head(self, base_model, symbol_id, config):
        """Create Transformer model with symbol-specific head from scratch"""
        from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Embedding, Concatenate, Add
        from tensorflow.keras.models import Model
        
        # Get input shapes from base model
        try:
            # Base model has two inputs: [feature_input, symbol_input]
            feature_shape = base_model.inputs[0].shape[1:]  # Remove batch dimension
            symbol_shape = base_model.inputs[1].shape[1:]   # Remove batch dimension
            
            # Convert TensorShape to tuple if needed
            if hasattr(feature_shape, 'as_list'):
                feature_shape = tuple(feature_shape.as_list())
            elif not isinstance(feature_shape, tuple):
                feature_shape = tuple(feature_shape)
                
            if hasattr(symbol_shape, 'as_list'):
                symbol_shape = tuple(symbol_shape.as_list())
            elif not isinstance(symbol_shape, tuple):
                symbol_shape = tuple(symbol_shape)
                
        except Exception as e:
            # Default fallback for Transformer
            feature_shape = (60, 5)  # sequence_length, features
            symbol_shape = ()        # symbol input shape
        
        # Create fresh input layers (matching base model)
        feature_input = Input(shape=feature_shape, name='feature_input')
        symbol_input = Input(shape=symbol_shape, dtype=tf.int32, name='symbol_input')
        
        # Symbol embedding (matching base model)
        symbol_embedding = Embedding(
            input_dim=self.num_symbols,
            output_dim=self.symbol_embedding_dim,
            name='symbol_embedding'
        )(symbol_input)
        
        # Recreate Transformer layers with same architecture as base model
        x = feature_input
        
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=2,
            key_dim=feature_shape[-1] // 2,
            name='multi_head_attention_1'
        )(x, x)
        
        # Add & Norm
        x = Add(name='add_1')([x, attention])
        x = LayerNormalization(name='layer_norm_1')(x)
        
        # Dropout
        x = Dropout(config.get('dropout', 0.1), name='dropout_1')(x)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D(name='global_average_pooling')(x)
        
        # Combine transformer output with symbol embedding (matching base model)
        combined = Concatenate(name='feature_symbol_concat')([pooled, symbol_embedding])
        
        # Add symbol-specific head
        symbol_dense1 = Dense(
            units=32,
            activation='relu',
            name=f'symbol_{symbol_id}_dense_1'
        )(combined)
        
        symbol_dropout = Dropout(
            config.get('dropout', 0.2),
            name=f'symbol_{symbol_id}_dropout'
        )(symbol_dense1)
        
        symbol_output = Dense(
            1,
            activation='sigmoid',
            name=f'symbol_{symbol_id}_output'
        )(symbol_dropout)
        
        # Create the model
        symbol_model = Model(
            inputs=[feature_input, symbol_input],
            outputs=symbol_output,
            name=f'symbol_{symbol_id}_transformer_model'
        )
        
        # Copy weights from base model (up to the feature extraction part)
        self._copy_base_weights(base_model, symbol_model, 'transformer')
        
        return symbol_model
    
    def _copy_base_weights(self, base_model, symbol_model, architecture):
        """Copy weights from base model to symbol model for the shared layers"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Define which layers to copy based on architecture
            if architecture == 'lstm':
                layers_to_copy = ['lstm_1', 'lstm_2', 'symbol_embedding']
            elif architecture == 'cnn':
                layers_to_copy = ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'flatten', 'symbol_embedding']
            elif architecture == 'transformer':
                layers_to_copy = ['multi_head_attention_1', 'layer_norm_1', 'global_average_pooling', 'symbol_embedding']
            else:
                logger.warning(f"Unknown architecture {architecture}, skipping weight copying")
                return
            
            # Copy weights for matching layers
            for layer_name in layers_to_copy:
                base_layer = None
                symbol_layer = None
                
                # Find layers in both models
                for layer in base_model.layers:
                    if layer_name in layer.name:
                        base_layer = layer
                        break
                
                for layer in symbol_model.layers:
                    if layer_name in layer.name:
                        symbol_layer = layer
                        break
                
                # Copy weights if both layers exist and have weights
                if base_layer and symbol_layer and base_layer.get_weights():
                    try:
                        symbol_layer.set_weights(base_layer.get_weights())
                        logger.info(f"✓ Copied weights for layer: {layer_name}")
                    except Exception as e:
                        logger.warning(f"Could not copy weights for layer {layer_name}: {e}")
                        
        except Exception as e:
            logger.warning(f"Weight copying failed: {e}")
            logger.info("Symbol model will be trained from scratch")

    def _get_feature_layer_for_architecture(self, base_model):
        """
        Get the feature extraction layer for different architectures.
        This method finds the actual layer that produces architecture-specific features
        and returns its output to maintain proper connectivity.
        
        Args:
            base_model: The base universal model
            
        Returns:
            The feature layer output tensor
        """
        logger.info(f"Feature extraction for model: {base_model.name}")
        logger.info("Available layers in model:")
        for i, layer in enumerate(base_model.layers):
            logger.info(f"  Layer {i}: {layer.name} ({layer.__class__.__name__})")
        
        # Determine architecture from model name
        model_name = base_model.name.lower()
        
        if 'lstm' in model_name:
            # For LSTM models, find the 'lstm_2' layer output
            logger.info("Detected LSTM architecture - searching for lstm_2 layer")
            for layer in base_model.layers:
                if layer.name == 'lstm_2':
                    logger.info(f"✓ Found target LSTM layer: {layer.name} ({layer.__class__.__name__}) - Output Shape: {getattr(layer, 'output_shape', 'N/A')}")
                    return layer.output  # Return the layer's output tensor, not layer.input[0]
            
            logger.error("✗ Could not find lstm_2 layer in LSTM model")
            raise ValueError("LSTM model missing expected lstm_2 layer")
            
        elif 'cnn' in model_name:
            # For CNN models, find the 'flatten' layer output
            logger.info("Detected CNN architecture - searching for flatten layer")
            for layer in base_model.layers:
                if layer.name == 'flatten':
                    logger.info(f"✓ Found target CNN layer: {layer.name} ({layer.__class__.__name__}) - Output Shape: {getattr(layer, 'output_shape', 'N/A')}")
                    return layer.output  # Return the layer's output tensor, not layer.input[0]
            
            logger.error("✗ Could not find flatten layer in CNN model")
            raise ValueError("CNN model missing expected flatten layer")
            
        elif 'transformer' in model_name:
            # For Transformer models, find the 'global_average_pooling' layer output
            logger.info("Detected Transformer architecture - searching for global_average_pooling layer")
            for layer in base_model.layers:
                if layer.name == 'global_average_pooling':
                    logger.info(f"✓ Found target Transformer layer: {layer.name} ({layer.__class__.__name__}) - Output Shape: {getattr(layer, 'output_shape', 'N/A')}")
                    return layer.output  # Return the layer's output tensor, not layer.input[0]
            
            logger.error("✗ Could not find global_average_pooling layer in Transformer model")
            raise ValueError("Transformer model missing expected global_average_pooling layer")
            
        else:
            logger.error(f"Unknown architecture in model name: {base_model.name}")
            raise ValueError(f"Cannot determine architecture from model name: {base_model.name}")

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

    def create_universal_xgboost(self, feature_dim: int, config: Dict, model_name: str = "universal_xgboost") -> Any:
        """
        Create XGBoost model optimized for trading aggregated features.
        """
        import xgboost as xgb
        
        logger.info(f"Creating XGBoost model with {feature_dim} aggregated features")
        
        model = xgb.XGBClassifier(
            # Core parameters
            n_estimators=config.get('n_estimators', 1000),
            max_depth=config.get('max_depth', 7),
            learning_rate=config.get('learning_rate', 0.15),
            
            # Regularization
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            reg_alpha=config.get('reg_alpha', 0.1),
            reg_lambda=config.get('reg_lambda', 0.1),
            
            # Performance
            n_jobs=-1,
            random_state=42,
            
            # Trading-specific
            objective='binary:logistic',
            tree_method='hist'
        )
        
        logger.info(f"Created XGBoost model: {model.n_estimators} trees")
        return model

    def create_universal_random_forest(self, feature_dim: int, config: Dict, model_name: str = "universal_random_forest") -> Any:
        """
        Create Random Forest model optimized for trading aggregated features.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info(f"Creating Random Forest model with {feature_dim} aggregated features")
        
        model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 500),
            max_depth=config.get('max_depth', 12),
            min_samples_split=config.get('min_samples_split', 10),
            min_samples_leaf=config.get('min_samples_leaf', 5),
            max_features=config.get('max_features', 'sqrt'),
            bootstrap=config.get('bootstrap', True),
            
            # Performance
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        logger.info(f"Created Random Forest: {model.n_estimators} trees")
        return model

    def create_universal_svm(self, feature_dim: int, config: Dict, model_name: str = "universal_svm") -> Any:
        """
        Create SVM model optimized for trading aggregated features.
        """
        from sklearn.svm import SVC
        
        logger.info(f"Creating SVM model with {feature_dim} aggregated features")
        
        # Use linear kernel for large datasets to avoid hanging
        # RBF kernel can be very slow on datasets with many features
        kernel = config.get('kernel', 'linear' if feature_dim > 20 else 'rbf')
        
        model = SVC(
            kernel=kernel,
            C=config.get('C', 0.1),  # Lower C for faster training
            gamma=config.get('gamma', 'scale'),
            class_weight=config.get('class_weight', 'balanced'),
            
            # Enable probability estimates
            probability=True,
            cache_size=2000,  # Increased cache for better performance
            max_iter=1000,    # Limit iterations to prevent hanging
            random_state=42
        )
        
        logger.info(f"Created SVM: kernel={model.kernel}, C={model.C}, max_iter={model.max_iter}")
        return model

    def create_ensemble_model(self, feature_dim: int, config: Dict, model_name: str = "universal_ensemble") -> Dict:
        """
        Create ensemble combining XGBoost, Random Forest, and SVM.
        """
        logger.info(f"Creating ensemble model with {feature_dim} aggregated features")
        
        # Create individual models
        xgb_model = self.create_universal_xgboost(feature_dim, config.get('xgboost', {}))
        rf_model = self.create_universal_random_forest(feature_dim, config.get('random_forest', {}))
        svm_model = self.create_universal_svm(feature_dim, config.get('svm', {}))
        
        ensemble = {
            'models': {
                'xgboost': xgb_model,
                'random_forest': rf_model,
                'svm': svm_model
            },
            'weights': {
                'xgboost': config.get('xgb_weight', 0.45),
                'random_forest': config.get('rf_weight', 0.35),
                'svm': config.get('svm_weight', 0.20)
            },
            'feature_dim': feature_dim,
            'name': model_name
        }
        
        logger.info(f"Created ensemble with weights: {ensemble['weights']}")
        return ensemble

    def save_statistical_model(self, model, model_path):
        """Save statistical models using joblib."""
        from pathlib import Path
        import joblib
        import json
        
        try:
            model_path = Path(model_path)
            
            if isinstance(model, dict) and 'models' in model:
                # Ensemble model - use standardized directory name
                ensemble_dir = model_path.parent / "ensemble_base_ensemble"
                ensemble_dir.mkdir(exist_ok=True)
                
                for model_name, individual_model in model['models'].items():
                    individual_path = ensemble_dir / f"{model_name}.joblib"
                    joblib.dump(individual_model, individual_path)
                
                # Save ensemble configuration - prioritize selected features from feature selection
                selected_feature_count = model.get('selected_feature_count', model['feature_dim'])
                selected_feature_columns = model.get('selected_feature_columns', [])
                
                # Log validation information
                logger.info(f"Saving ensemble config: selected_feature_count={selected_feature_count}, feature_dim={model['feature_dim']}")
                if selected_feature_count != model['feature_dim']:
                    logger.warning(f"Feature count mismatch in ensemble config: selected={selected_feature_count}, actual={model['feature_dim']}")
                
                ensemble_config = {
                    'weights': model['weights'],
                    'feature_dim': selected_feature_count,  # Use selected feature count as the authoritative dimension
                    'name': model['name'],
                    'selected_feature_count': selected_feature_count,
                    'selected_feature_columns': selected_feature_columns
                }
                config_path = ensemble_dir / "ensemble_config.json"
                with open(config_path, 'w') as f:
                    json.dump(ensemble_config, f)
                    
                logger.info(f"Saved ensemble model to {ensemble_dir}")
            else:
                # Individual statistical model
                joblib_path = str(model_path).replace('.h5', '.joblib')
                joblib.dump(model, joblib_path)
                logger.info(f"Saved statistical model to {joblib_path}")
                
        except Exception as e:
            logger.error(f"Failed to save statistical model: {e}")
            raise

    def load_statistical_model(self, model_path):
        """Load statistical models using joblib."""
        from pathlib import Path
        import joblib
        import json
        
        try:
            model_path = Path(model_path)
            
            # Check if it's an ensemble - use standardized directory name
            ensemble_dir = model_path.parent / "ensemble_base_ensemble"
            if ensemble_dir.exists():
                # Load ensemble
                models = {}
                for model_file in ensemble_dir.glob("*.joblib"):
                    model_name = model_file.stem
                    models[model_name] = joblib.load(model_file)
                
                # Load ensemble configuration
                config_path = ensemble_dir / "ensemble_config.json"
                with open(config_path, 'r') as f:
                    ensemble_config = json.load(f)
                
                ensemble = {
                    'models': models,
                    'weights': ensemble_config['weights'],
                    'feature_dim': ensemble_config['feature_dim'],
                    'name': ensemble_config['name']
                }
                
                logger.info(f"Loaded ensemble model from {ensemble_dir}")
                return ensemble
            else:
                # Individual statistical model
                joblib_path = str(model_path).replace('.h5', '.joblib')
                if Path(joblib_path).exists():
                    model = joblib.load(joblib_path)
                    logger.info(f"Loaded statistical model from {joblib_path}")
                    return model
                else:
                    raise FileNotFoundError(f"Statistical model not found at {joblib_path}")
                    
        except Exception as e:
            logger.error(f"Failed to load statistical model: {e}")
            raise