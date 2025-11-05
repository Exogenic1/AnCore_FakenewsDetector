"""
AnCore - Main Application
News Article Credibility Assessment and Fake News Detection using mBERT

This is the main entry point for the AnCore project that implements
a fake news detection system using multilingual BERT (mBERT).
"""

import os
import torch
import argparse
from datetime import datetime

from ancore_config import Config
from ancore_dataset import DataProcessor
from ancore_model import mBERTClassifier, CredibilityAssessor
from ancore_trainer import Trainer


class AnCore:
    """Main application class for AnCore fake news detection system"""
    
    def __init__(self):
        """Initialize AnCore system"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*60}")
        print("AnCore - News Article Credibility Assessment")
        print("Fake News Detection using mBERT")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"{'='*60}\n")
        
        # Create output directories
        Config.create_directories()
        
        self.model = None
        self.trainer = None
        self.data_processor = DataProcessor()
        
    def prepare_data(self):
        """Prepare training, validation, and test data"""
        print("Preparing data...")
        data_path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        train_loader, val_loader, test_loader, data_splits = self.data_processor.prepare_data(data_path)
        
        return train_loader, val_loader, test_loader, data_splits
    
    def initialize_model(self):
        """Initialize mBERT model"""
        print("\nInitializing mBERT model...")
        self.model = mBERTClassifier(num_labels=Config.NUM_LABELS)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def train(self, train_loader, val_loader, test_loader):
        """Train the model"""
        if self.model is None:
            self.initialize_model()
        
        print("\nInitializing trainer...")
        self.trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device
        )
        
        # Train the model
        history = self.trainer.train()
        
        # Plot training history
        self.trainer.plot_training_history(save_path='training_history.png')
        
        return history
    
    def evaluate(self):
        """Evaluate the trained model"""
        if self.trainer is None:
            raise ValueError("Model must be trained or loaded before evaluation")
        
        # Load best model
        if os.path.exists(os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pt')):
            print("\nLoading best model...")
            self.trainer.load_model('best_model.pt')
        
        # Evaluate on test set
        results = self.trainer.evaluate()
        
        # Plot confusion matrix
        self.trainer.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path='confusion_matrix.png'
        )
        
        # Save results
        self.save_results(results)
        
        return results
    
    def predict(self, text):
        """
        Predict if a single article is fake or real
        
        Args:
            text: News article text
            
        Returns:
            Dictionary with prediction and credibility assessment
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Tokenize the text
        encoding = self.data_processor.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Create credibility assessor
        assessor = CredibilityAssessor(self.model, self.device)
        
        # Get prediction
        results = assessor.assess_credibility(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        return results[0]
    
    def save_results(self, results):
        """Save evaluation results to file"""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(Config.RESULTS_DIR, f'results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'confusion_matrix': results['confusion_matrix'],
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
    
    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        try:
            # Prepare data
            train_loader, val_loader, test_loader, _ = self.prepare_data()
            
            # Train model
            history = self.train(train_loader, val_loader, test_loader)
            
            # Evaluate model
            results = self.evaluate()
            
            print("\n" + "="*60)
            print("AnCore Pipeline Complete!")
            print("="*60)
            print(f"Final Test Accuracy: {results['accuracy']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            print("="*60 + "\n")
            
            return history, results
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
    
    def interactive_mode(self):
        """Interactive mode for testing predictions"""
        print("\n" + "="*60)
        print("AnCore Interactive Mode")
        print("Enter news articles to check credibility")
        print("Type 'quit' to exit")
        print("="*60 + "\n")
        
        # Load best model if available
        if os.path.exists(os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pt')):
            print("Loading model...")
            if self.model is None:
                self.initialize_model()
            if self.trainer is None:
                # Create dummy loaders for trainer initialization
                from torch.utils.data import DataLoader, TensorDataset
                dummy_loader = DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)
                self.trainer = Trainer(self.model, dummy_loader, dummy_loader, dummy_loader, self.device)
            self.trainer.load_model('best_model.pt')
            print("Model loaded successfully!\n")
        else:
            print("No trained model found. Please train the model first.\n")
            return
        
        while True:
            print("-" * 60)
            article = input("\nEnter article text (or 'quit' to exit):\n> ")
            
            if article.lower() == 'quit':
                print("\nExiting interactive mode...")
                break
            
            if not article.strip():
                print("Please enter some text.")
                continue
            
            try:
                result = self.predict(article)
                
                print("\n" + "="*60)
                print("CREDIBILITY ASSESSMENT")
                print("="*60)
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%} ({result['confidence_level']})")
                print(f"\nProbability Breakdown:")
                print(f"  Real News: {result['probability_real']:.2%}")
                print(f"  Fake News: {result['probability_fake']:.2%}")
                print(f"\nCredibility Score: {result['probability_real']*100:.1f}/100")
                print("="*60)
                
            except Exception as e:
                print(f"\nError processing article: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AnCore - Fake News Detection using mBERT')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'predict', 'interactive'],
                       help='Mode to run: train, evaluate, predict, or interactive')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to predict (for predict mode)')
    
    args = parser.parse_args()
    
    # Create AnCore instance
    ancore = AnCore()
    
    if args.mode == 'train':
        # Run full training and evaluation pipeline
        ancore.run_full_pipeline()
        
    elif args.mode == 'evaluate':
        # Only evaluate existing model
        ancore.initialize_model()
        train_loader, val_loader, test_loader, _ = ancore.prepare_data()
        ancore.trainer = Trainer(ancore.model, train_loader, val_loader, test_loader, ancore.device)
        ancore.evaluate()
        
    elif args.mode == 'predict':
        # Predict single article
        if args.text is None:
            print("Error: --text argument required for predict mode")
            return
        
        ancore.initialize_model()
        # Load best model
        train_loader, val_loader, test_loader, _ = ancore.prepare_data()
        ancore.trainer = Trainer(ancore.model, train_loader, val_loader, test_loader, ancore.device)
        
        if os.path.exists(os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pt')):
            ancore.trainer.load_model('best_model.pt')
            result = ancore.predict(args.text)
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Credibility Score: {result['probability_real']*100:.1f}/100")
        else:
            print("No trained model found. Please train the model first.")
    
    elif args.mode == 'interactive':
        # Interactive mode
        ancore.interactive_mode()


if __name__ == "__main__":
    main()
