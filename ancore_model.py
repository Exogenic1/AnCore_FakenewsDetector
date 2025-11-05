"""
AnCore - mBERT Model Implementation
Implements the multilingual BERT model for fake news detection
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from ancore_config import Config


class mBERTClassifier(nn.Module):
    """
    mBERT-based classifier for fake news detection
    Uses multilingual BERT as the backbone with a classification head
    """
    
    def __init__(self, num_labels=2, dropout_prob=0.3):
        """
        Args:
            num_labels: Number of output labels (2 for binary classification)
            dropout_prob: Dropout probability for regularization
        """
        super(mBERTClassifier, self).__init__()
        
        # Load pretrained mBERT model
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        
        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize weights for classifier
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the classification layer"""
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            
        Returns:
            logits: Raw prediction scores
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from classifier
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_proba(self, input_ids, attention_mask):
        """
        Get probability predictions
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            
        Returns:
            probabilities: Softmax probabilities
        """
        logits = self.forward(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def freeze_bert_encoder(self):
        """Freeze BERT encoder parameters (for fine-tuning only classifier)"""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        """Unfreeze BERT encoder parameters (for full fine-tuning)"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def freeze_bert_layers(self, num_layers):
        """
        Freeze first N layers of BERT
        
        Args:
            num_layers: Number of layers to freeze from the bottom
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False


class CredibilityAssessor:
    """
    Assesses the credibility of news articles using the trained model
    """
    
    def __init__(self, model, device):
        """
        Args:
            model: Trained mBERT classifier
            device: torch device (cuda or cpu)
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def assess_credibility(self, input_ids, attention_mask):
        """
        Assess the credibility of an article
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            
        Returns:
            Dictionary with prediction, probability, and confidence
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get probabilities
            probabilities = self.model.predict_proba(input_ids, attention_mask)
            
            # Get predictions
            predictions = torch.argmax(probabilities, dim=1)
            
            # Get confidence scores
            confidence_scores = torch.max(probabilities, dim=1)[0]
            
            results = []
            for i in range(len(predictions)):
                pred = int(predictions[i].item())
                conf = float(confidence_scores[i].item())
                prob_real = float(probabilities[i][0].item())
                prob_fake = float(probabilities[i][1].item())
                
                # Determine confidence level
                if conf >= Config.HIGH_CONFIDENCE_THRESHOLD:
                    confidence_level = "High"
                elif conf >= Config.LOW_CONFIDENCE_THRESHOLD:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                results.append({
                    'prediction': Config.LABELS[pred],
                    'prediction_id': pred,
                    'confidence': conf,
                    'confidence_level': confidence_level,
                    'probability_real': prob_real,
                    'probability_fake': prob_fake
                })
            
            return results
    
    def get_credibility_score(self, probability_real):
        """
        Convert probability to credibility score (0-100)
        
        Args:
            probability_real: Probability of being real news
            
        Returns:
            Credibility score from 0 to 100
        """
        return probability_real * 100


def test_model():
    """Test the model architecture"""
    print("Testing mBERT Classifier...")
    
    # Create model
    model = mBERTClassifier(num_labels=Config.NUM_LABELS)
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    # Create dummy input
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    print(f"Output shape: {logits.shape}")
    
    # Test probability prediction
    probs = model.predict_proba(input_ids, attention_mask)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs}")
    
    # Test credibility assessor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    assessor = CredibilityAssessor(model, device)
    
    results = assessor.assess_credibility(input_ids, attention_mask)
    print(f"\nSample credibility assessment:")
    for i, result in enumerate(results):
        print(f"\nArticle {i+1}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    print("\n✓ Model test completed successfully!")


if __name__ == "__main__":
    test_model()
