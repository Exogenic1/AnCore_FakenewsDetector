"""
AnCore - Quick Start Demo
A simple demo script to quickly test the AnCore system
"""

import os
import torch
from ancore_config import Config
from ancore_dataset import DataProcessor
from ancore_model import mBERTClassifier
from ancore_trainer import Trainer


def quick_test():
    """Quick test with minimal training for demonstration"""
    print("\n" + "="*70)
    print("AnCore Quick Start Demo")
    print("="*70)
    
    # Check if data file exists
    data_path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please ensure fakenews/full.csv exists in the project directory.")
        return
    
    print("\n✓ Data file found")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Quick training configuration (reduced for demo)
    print("\n📝 Configuration:")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Max Length: {Config.MAX_LENGTH}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Epochs: {Config.NUM_EPOCHS}")
    
    # Prepare data
    print("\n📊 Preparing data...")
    processor = DataProcessor()
    train_loader, val_loader, test_loader, _ = processor.prepare_data(data_path)
    print("✓ Data preparation complete")
    
    # Show sample
    print("\n📄 Sample data:")
    sample_batch = next(iter(train_loader))
    print(f"   Batch size: {sample_batch['input_ids'].size(0)}")
    print(f"   Sequence length: {sample_batch['input_ids'].size(1)}")
    print(f"   Labels: {sample_batch['label'].tolist()}")
    
    # Initialize model
    print("\n🤖 Initializing mBERT model...")
    model = mBERTClassifier(num_labels=Config.NUM_LABELS)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("Demo setup complete!")
    print("="*70)
    print("\n📌 Next steps:")
    print("   1. To train the model: python ancore_main.py --mode train")
    print("   2. To test predictions: python ancore_main.py --mode interactive")
    print("   3. To evaluate model: python ancore_main.py --mode evaluate")
    print("\n" + "="*70)


def show_sample_articles():
    """Show sample articles from the dataset"""
    import pandas as pd
    
    print("\n" + "="*70)
    print("Sample Articles from Dataset")
    print("="*70)
    
    data_path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
    
    df = pd.read_csv(data_path)
    
    # Show a real news sample
    real_sample = df[df['label'] == 0].iloc[0]
    print("\n📰 REAL NEWS SAMPLE:")
    print("-" * 70)
    print(real_sample['article'][:300] + "...")
    
    # Show a fake news sample
    fake_sample = df[df['label'] == 1].iloc[0]
    print("\n❌ FAKE NEWS SAMPLE:")
    print("-" * 70)
    print(fake_sample['article'][:300] + "...")
    
    print("\n" + "="*70)
    print(f"Dataset contains {len(df)} articles:")
    print(f"   Real news: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
    print(f"   Fake news: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
    print("="*70)


def main():
    """Main demo function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--samples':
        show_sample_articles()
    else:
        quick_test()
        print("\n💡 Tip: Run 'python demo.py --samples' to see sample articles")


if __name__ == "__main__":
    main()
