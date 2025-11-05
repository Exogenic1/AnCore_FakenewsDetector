"""
AnCore - Dataset Processing Module
Handles data loading, preprocessing, and tokenization for mBERT
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from ancore_config import Config


class FakeNewsDataset(Dataset):
    """Custom Dataset for Fake News Detection"""
    
    def __init__(self, articles, labels, tokenizer, max_length):
        """
        Args:
            articles: List of news articles
            labels: List of labels (0=Real, 1=Fake)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.articles = articles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        label = self.labels[idx]
        
        # Tokenize the article
        encoding = self.tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """Process and prepare data for training"""
    
    def __init__(self, config=None):
        """
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with articles and labels
        """
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} articles")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return df
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame with articles and labels
            
        Returns:
            Train, validation, and test DataFrames
        """
        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SPLIT,
            random_state=Config.RANDOM_SEED,
            stratify=df['label']
        )
        
        # Second split: train and validation
        val_size = Config.VAL_SPLIT / (Config.TRAIN_SPLIT + Config.VAL_SPLIT)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=Config.RANDOM_SEED,
            stratify=train_val_df['label']
        )
        
        print(f"\nData split:")
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_data_loader(self, df, batch_size, shuffle=True):
        """
        Create DataLoader for the dataset
        
        Args:
            df: DataFrame with articles and labels
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader object
        """
        dataset = FakeNewsDataset(
            articles=df['article'].values,
            labels=df['label'].values,
            tokenizer=self.tokenizer,
            max_length=Config.MAX_LENGTH
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    def prepare_data(self, data_path):
        """
        Complete data preparation pipeline
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Train, validation, and test DataLoaders
        """
        # Load data
        df = self.load_data(data_path)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Create DataLoaders
        train_loader = self.create_data_loader(
            train_df, 
            Config.BATCH_SIZE, 
            shuffle=True
        )
        val_loader = self.create_data_loader(
            val_df, 
            Config.BATCH_SIZE, 
            shuffle=False
        )
        test_loader = self.create_data_loader(
            test_df, 
            Config.BATCH_SIZE, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader, (train_df, val_df, test_df)


def main():
    """Test the data processing module"""
    import os
    
    data_path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    processor = DataProcessor()
    train_loader, val_loader, test_loader, _ = processor.prepare_data(data_path)
    
    print("\n=== Data Processing Complete ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Show a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shape:")
    print(f"Input IDs: {sample_batch['input_ids'].shape}")
    print(f"Attention Mask: {sample_batch['attention_mask'].shape}")
    print(f"Labels: {sample_batch['label'].shape}")


if __name__ == "__main__":
    main()
