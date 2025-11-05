"""
AnCore - Data Exploration Utility
Analyze and visualize the fake news dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from ancore_config import Config


def load_dataset():
    """Load the dataset"""
    data_path = os.path.join(Config.DATA_DIR, Config.DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)


def basic_statistics(df):
    """Display basic statistics about the dataset"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal articles: {len(df)}")
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        label_name = Config.LABELS[label]
        percentage = (count / len(df)) * 100
        print(f"  {label_name} (label={label}): {count} ({percentage:.1f}%)")
    
    # Article length statistics
    df['article_length'] = df['article'].str.len()
    df['word_count'] = df['article'].str.split().str.len()
    
    print(f"\nArticle length statistics (characters):")
    print(f"  Mean: {df['article_length'].mean():.0f}")
    print(f"  Median: {df['article_length'].median():.0f}")
    print(f"  Min: {df['article_length'].min()}")
    print(f"  Max: {df['article_length'].max()}")
    
    print(f"\nWord count statistics:")
    print(f"  Mean: {df['word_count'].mean():.0f}")
    print(f"  Median: {df['word_count'].median():.0f}")
    print(f"  Min: {df['word_count'].min()}")
    print(f"  Max: {df['word_count'].max()}")


def visualize_distribution(df):
    """Create visualizations of the data distribution"""
    Config.create_directories()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Label distribution
    label_counts = df['label'].value_counts()
    labels = [Config.LABELS[i] for i in label_counts.index]
    axes[0, 0].bar(labels, label_counts.values, color=['green', 'red'])
    axes[0, 0].set_title('Distribution of Real vs Fake News', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(label_counts.values):
        axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    # Article length distribution
    df['article_length'] = df['article'].str.len()
    axes[0, 1].hist(df['article_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Distribution of Article Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Characters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['article_length'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["article_length"].mean():.0f}')
    axes[0, 1].legend()
    
    # Word count distribution
    df['word_count'] = df['article'].str.split().str.len()
    axes[1, 0].hist(df['word_count'], bins=50, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Distribution of Word Counts', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Words')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df['word_count'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {df["word_count"].mean():.0f}')
    axes[1, 0].legend()
    
    # Length comparison by label
    real_lengths = df[df['label'] == 0]['article_length']
    fake_lengths = df[df['label'] == 1]['article_length']
    
    box_data = [real_lengths, fake_lengths]
    bp = axes[1, 1].boxplot(box_data, labels=['Real News', 'Fake News'],
                             patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    axes[1, 1].set_title('Article Length by Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Characters')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(Config.RESULTS_DIR, 'dataset_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.show()


def show_samples(df, num_samples=3):
    """Display sample articles from each category"""
    print("\n" + "="*70)
    print("SAMPLE ARTICLES")
    print("="*70)
    
    for label in [0, 1]:
        label_name = Config.LABELS[label]
        print(f"\n{'='*70}")
        print(f"{label_name.upper()} SAMPLES")
        print(f"{'='*70}")
        
        samples = df[df['label'] == label].sample(min(num_samples, len(df[df['label'] == label])))
        
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            print(f"\nSample {idx}:")
            print("-" * 70)
            article_text = row['article'][:400]  # First 400 characters
            print(f"{article_text}...")
            print(f"\nLength: {len(row['article'])} characters, "
                  f"{len(row['article'].split())} words")


def analyze_vocabulary(df, top_n=20):
    """Analyze most common words in real vs fake news"""
    print("\n" + "="*70)
    print("VOCABULARY ANALYSIS")
    print("="*70)
    
    # Common Filipino stopwords (basic list)
    stopwords = {'ang', 'ng', 'sa', 'na', 'at', 'ay', 'mga', 'si', 'ni', 
                 'ko', 'mo', 'ka', 'ako', 'ikaw', 'siya', 'kami', 'kayo', 
                 'sila', 'nang', 'pa', 'po', 'rin', 'din', 'daw', 'raw'}
    
    for label in [0, 1]:
        label_name = Config.LABELS[label]
        articles = df[df['label'] == label]['article']
        
        # Combine all articles and split into words
        all_words = ' '.join(articles).lower().split()
        
        # Filter out stopwords and short words
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 2]
        
        # Count words
        word_counts = Counter(filtered_words)
        most_common = word_counts.most_common(top_n)
        
        print(f"\n{label_name} - Top {top_n} words:")
        print("-" * 70)
        for word, count in most_common:
            print(f"  {word:20s}: {count:5d}")


def compare_categories(df):
    """Compare statistics between real and fake news"""
    print("\n" + "="*70)
    print("CATEGORY COMPARISON")
    print("="*70)
    
    df['article_length'] = df['article'].str.len()
    df['word_count'] = df['article'].str.split().str.len()
    
    for label in [0, 1]:
        label_name = Config.LABELS[label]
        subset = df[df['label'] == label]
        
        print(f"\n{label_name}:")
        print(f"  Count: {len(subset)}")
        print(f"  Avg length: {subset['article_length'].mean():.0f} characters")
        print(f"  Avg words: {subset['word_count'].mean():.0f}")
        print(f"  Median length: {subset['article_length'].median():.0f} characters")
        print(f"  Median words: {subset['word_count'].median():.0f}")


def main():
    """Main exploration function"""
    try:
        print("\n" + "="*70)
        print("AnCore Dataset Exploration")
        print("="*70)
        
        # Load data
        print("\nLoading dataset...")
        df = load_dataset()
        print(f"✓ Loaded {len(df)} articles")
        
        # Basic statistics
        basic_statistics(df)
        
        # Category comparison
        compare_categories(df)
        
        # Show samples
        show_samples(df, num_samples=2)
        
        # Vocabulary analysis
        analyze_vocabulary(df, top_n=15)
        
        # Visualizations
        print("\n" + "="*70)
        print("Generating visualizations...")
        visualize_distribution(df)
        
        print("\n" + "="*70)
        print("Exploration Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
