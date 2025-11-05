

---
annotations_creators:
- expert-generated
language_creators:
- crowdsourced
language:
- tl
license:
- unknown
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- fact-checking
paperswithcode_id: fake-news-filipino-dataset
pretty_name: Fake News Filipino
dataset_info:
  features:
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
  - name: article
    dtype: string
  splits:
  - name: train
    num_bytes: 3623685
    num_examples: 3206
  download_size: 1313458
  dataset_size: 3623685
---

# AnCore - News Article Credibility Assessment and Fake News Detection

## Project Overview

**AnCore** is a sophisticated fake news detection system using multilingual BERT (mBERT) for assessing the credibility of Filipino news articles. This project implements state-of-the-art natural language processing techniques to classify news articles as either real or fake with high accuracy.

## Features

- **mBERT-based Classification**: Utilizes Google's multilingual BERT for robust language understanding
- **Credibility Assessment**: Provides confidence scores and probability distributions
- **Interactive Prediction**: Test individual articles through an interactive interface
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Visualization**: Training curves and confusion matrix plots
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Model Checkpointing**: Saves best performing model automatically

## Project Structure

```
Fakenews/
├── ancore_main.py          # Main application entry point
├── ancore_config.py        # Configuration settings
├── ancore_dataset.py       # Data processing and loading
├── ancore_model.py         # mBERT model architecture
├── ancore_trainer.py       # Training and evaluation logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── fakenews/
│   └── full.csv           # Dataset
└── output/                # Generated outputs
    ├── models/            # Saved model checkpoints
    ├── results/           # Evaluation results
    └── logs/              # Training logs
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd d:\Fakenews
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Verify dataset:**
Ensure `fakenews/full.csv` exists in the project directory.

## Usage

### 1. Training the Model

Run the complete training and evaluation pipeline:

```bash
python ancore_main.py --mode train
```

This will:
- Load and preprocess the dataset
- Split data into train/validation/test sets
- Initialize the mBERT model
- Train the model with early stopping
- Evaluate on the test set
- Generate visualization plots
- Save the best model

### 2. Evaluating a Trained Model

Evaluate an existing trained model:

```bash
python ancore_main.py --mode evaluate
```

### 3. Predicting Single Articles

Predict credibility of a specific article:

```bash
python ancore_main.py --mode predict --text "Your news article text here"
```

### 4. Interactive Mode

Launch interactive mode for testing multiple articles:

```bash
python ancore_main.py --mode interactive
```

In interactive mode, you can:
- Enter news articles one at a time
- Get instant credibility assessments
- See probability breakdowns
- Test multiple articles in succession

## Configuration

Modify `ancore_config.py` to adjust:

- **Model Parameters**: Learning rate, batch size, epochs
- **Data Split**: Train/validation/test ratios
- **Thresholds**: Confidence levels for credibility assessment
- **Paths**: Output directories and file locations

Key configuration options:

```python
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
```

## Model Architecture

The AnCore system uses a fine-tuned mBERT model with:

- **Base Model**: bert-base-multilingual-cased (110M parameters)
- **Classification Head**: Linear layer for binary classification
- **Regularization**: Dropout (0.3) to prevent overfitting
- **Optimization**: AdamW optimizer with linear warmup schedule

## Output Files

### Model Checkpoints
- `output/models/best_model.pt` - Best performing model on validation set

### Visualizations
- `output/results/training_history.png` - Training/validation curves
- `output/results/confusion_matrix.png` - Confusion matrix heatmap

### Results
- `output/results/results_YYYYMMDD_HHMMSS.json` - Evaluation metrics

## Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Dataset

The project uses the **Fake News Filipino Dataset**:

- **Size**: 3,206 expertly-labeled news articles
- **Balance**: 50% real news, 50% fake news
- **Language**: Filipino (Tagalog)
- **Citation**: Cruz et al., 2020, LREC

## Credibility Assessment

Each prediction includes:

1. **Label**: Real News or Fake News
2. **Confidence**: Probability of the prediction (0-1)
3. **Confidence Level**: High/Medium/Low based on thresholds
4. **Probability Breakdown**: Separate probabilities for real/fake
5. **Credibility Score**: 0-100 scale based on "real news" probability

## Examples

### Training Output
```
===================================================
Starting Training
===================================================

Epoch 1/5
--------------------------------------------------
Train Loss: 0.4523 | Train Acc: 0.7845
Val Loss: 0.3821 | Val Acc: 0.8234
✓ New best model saved! (Val Acc: 0.8234)
```

### Prediction Output
```
============================================================
CREDIBILITY ASSESSMENT
============================================================
Prediction: Real News
Confidence: 92.34% (High)

Probability Breakdown:
  Real News: 92.34%
  Fake News: 7.66%

Credibility Score: 92.3/100
============================================================
```

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `ancore_config.py`
- Reduce `MAX_LENGTH` to process shorter sequences

### Slow Training
- Enable GPU if available (CUDA)
- Reduce `NUM_EPOCHS` for faster experimentation
- Use a smaller batch size

### Data File Not Found
- Ensure `fakenews/full.csv` exists
- Check file path in `ancore_config.py`

## Requirements

See `requirements.txt` for complete list. Key dependencies:

- torch >= 2.0.0
- transformers >= 4.30.0
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0

## References

1. Cruz, J. C. B., Tan, J. A., & Cheng, C. (2020). Localization of Fake News Detection via Multitask Transfer Learning. In Proceedings of The 12th Language Resources and Evaluation Conference (pp. 2596-2604).

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

## License

This project uses the Fake News Filipino dataset. Please refer to the original dataset repository for licensing information.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the project repository.

---

**AnCore** - Empowering truth through AI-driven credibility assessment.

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [Fake News Filipino homepage](https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks)
- **Repository:** [Fake News Filipino repository](https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks)
- **Paper:** [LREC 2020 paper](http://www.lrec-conf.org/proceedings/lrec2020/index.html)
- **Leaderboard:**
- **Point of Contact:** [Jan Christian Cruz](mailto:jan_christian_cruz@dlsu.edu.ph)

### Dataset Summary

Low-Resource Fake News Detection Corpora in Filipino. The first of its kind. Contains 3,206 expertly-labeled news samples, half of which are real and half of which are fake.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

The dataset is primarily in Filipino, with the addition of some English words commonly used in Filipino vernacular.

## Dataset Structure

### Data Instances

Sample data:
```
{
  "label": "0",
  "article": "Sa 8-pahinang desisyon, pinaboran ng Sandiganbayan First Division ang petition for Writ of Preliminary Attachment/Garnishment na inihain ng prosekusyon laban sa mambabatas."
}
```


### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

Fake news articles were sourced from online sites that were tagged as fake news sites by the non-profit independent media fact-checking organization Verafiles and the National Union of Journalists in the Philippines (NUJP). Real news articles were sourced from mainstream news websites in the Philippines, including Pilipino Star Ngayon, Abante, and Bandera.

### Curation Rationale

We remedy the lack of a proper, curated benchmark dataset for fake news detection in Filipino by constructing and producing what we call “Fake News Filipino.” 


### Source Data

#### Initial Data Collection and Normalization

We construct the dataset by scraping our source websites, encoding all characters into UTF-8. Preprocessing was light to keep information intact: we retain capitalization and punctuation, and do not correct any misspelled words.

#### Who are the source language producers?

Jan Christian Blaise Cruz, Julianne Agatha Tan, and Charibeth Cheng

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[Jan Christian Cruz](mailto:jan_christian_cruz@dlsu.edu.ph), Julianne Agatha Tan, and Charibeth Cheng

### Licensing Information

[More Information Needed]

### Citation Information

    @inproceedings{cruz2020localization,
      title={Localization of Fake News Detection via Multitask Transfer Learning},
      author={Cruz, Jan Christian Blaise and Tan, Julianne Agatha and Cheng, Charibeth},
      booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
      pages={2596--2604},
      year={2020}
    }

### Contributions

Thanks to [@anaerobeth](https://github.com/anaerobeth) for adding this dataset.