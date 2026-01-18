# Classical NLP vs Transformer Models

This project compares classical Natural Language Processing (NLP) techniques with transformer-based models for text classification.

## Objective
To analyze the performance, efficiency, and trade-offs between traditional NLP pipelines and modern transformer architectures on the same dataset.

## Approaches Compared

### Classical NLP
- Text preprocessing and cleaning
- Feature extraction using vectorization techniques (e.g. Bag of Words / TF-IDF)
- Machine learning classifiers (e.g. Logistic Regression, Naive Bayes)

### Transformer-Based NLP
- Pre-trained transformer models
- Fine-tuning for text classification
- Tokenization using model-specific tokenizers

## Evaluation
Both approaches are evaluated using standard classification metrics such as accuracy and F1-score, along with qualitative observations on training time and resource usage.

## Results
- Classical NLP provides faster training and interpretability
- Transformer models achieve higher performance at increased computational cost

## Folder Structure
- `data/` – Dataset used for training and evaluation
- `predictions/` – Model prediction outputs
- `comparison.ipynb` – Complete experimentation notebook

## Tools & Libraries
- Python
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Jupyter Notebook

## Notes
This project was created to build an intuitive understanding of when and why different NLP approaches should be used in real-world applications.
