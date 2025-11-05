"""
AnCore - News Article Credibility Assessment Configuration
Configuration file for the mBERT-based fake news detection system
"""

import os

class Config:
    """Configuration class for AnCore project"""
    
    # Model Configuration
    MODEL_NAME = 'bert-base-multilingual-cased'  # mBERT model
    MAX_LENGTH = 512  # Maximum sequence length for BERT
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data Configuration
    DATA_DIR = 'fakenews'
    DATA_FILE = 'full.csv'
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # Labels
    LABELS = {
        0: 'Real News',
        1: 'Fake News'
    }
    NUM_LABELS = len(LABELS)
    
    # Output Configuration
    OUTPUT_DIR = 'output'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Training Configuration
    EARLY_STOPPING_PATIENCE = 3
    SAVE_BEST_MODEL = True
    EVALUATION_STRATEGY = 'epoch'
    SAVE_STRATEGY = 'epoch'
    LOAD_BEST_MODEL_AT_END = True
    
    # Credibility Assessment Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.5
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
