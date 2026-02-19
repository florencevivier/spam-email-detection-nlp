# Spam Email Detection & Text Analysis

## Project Overview

This project aims to build a machine learning model capable of detecting spam emails and to perform an in-depth textual analysis of spam and regular (ham) messages.

The project combines:

* Exploratory Data Analysis (EDA)
* Text preprocessing and NLP techniques
* TF-IDF vectorization
* Logistic Regression classification
* Threshold tuning
* Topic modeling with LDA
* Semantic similarity analysis with word embeddings
* Named Entity Recognition (NER)

The workflow goes beyond simple classification and explores the structure, themes, and entities present in the dataset.


## Business Objective

The primary goal is to correctly classify emails as:

* **1 → Spam**
* **0 → Not Spam (Ham)**

From a practical perspective:

* **False Positives (ham classified as spam)** are highly problematic
  → Important emails may be lost.
* **False Negatives (spam classified as ham)** are less critical but still undesirable.

Therefore, the objective is to achieve strong overall performance while carefully tuning the decision threshold to reduce false positives.


## Dataset

The dataset contains **5,171 emails**, with the following columns:

* ID column (removed during preprocessing)
* Label (`ham` / `spam`)
* Text
* Encoded label (0/1)

After cleaning and duplicate removal:

* ~71% Ham emails
* ~29% Spam emails

The dataset is slightly imbalanced but manageable.


## Exploratory Data Analysis

Key findings:

* No missing values
* Duplicates were detected and removed
* Slight class imbalance (≈70/30 split)
* No significant difference in average email length between spam and ham
* Word clouds reveal strong lexical differences between spam and regular emails

Spam emails frequently contain:

* Commercial and promotional vocabulary
* Pharmaceutical-related terms
* Business-related terminology


## Text Preprocessing

The preprocessing pipeline includes:

* Lowercasing
* Punctuation removal
* Lemmatization (spaCy)
* Stopword removal (NLTK)
* Number removal

Two preprocessing pipelines were used:

* **TF-IDF pipeline** for classification
* **Bag-of-Words pipeline** for LDA topic modeling

Train-test split:

* 80% Training
* 20% Testing
* Stratified to preserve class distribution


## Model: Logistic Regression

TF-IDF vectorization was applied before training.

The chosen model:

```
LogisticRegression(class_weight="balanced")
```

Class balancing helps mitigate the slight class imbalance.


## Threshold Optimization

Default threshold: 0.5

To reduce false positives, the decision threshold was tuned between 0.5 and 1.0.

Optimal threshold identified: **0.7**

This improved business alignment by reducing ham emails misclassified as spam while maintaining strong F1 performance.


## Final Model Performance (Test Set)

| Metric          | Value                          |
| --------------- | ------------------------------ |
| F1 Score        | High (no overfitting observed) |
| False Negatives | Very low                       |
| False Positives | Reduced after threshold tuning |

The model shows:

* Strong generalization
* Stable train-test performance
* Effective balance between precision and recall


## Topic Modeling on Spam Emails (LDA)

To understand the structure of spam messages, Latent Dirichlet Allocation (LDA) was applied.

* Only spam emails were used
* Bag-of-Words representation
* Coherence score used to select optimal number of topics

Best number of topics: **4**
Coherence score: > 0.6

### Identified Topics

1. **Business & Finance**
2. **Health & Pharmaceuticals**
3. **Health & Research**
4. **Business Correspondence / Promotions**

Semantic similarity between topics was measured using pre-trained GloVe embeddings and cosine similarity.

Findings:

* Topics 0 and 3 are semantically close
* Topic 2 is clearly distinct from the others

This confirms heterogeneity within spam emails.


## Named Entity Recognition (NER) on Regular Emails

To analyze regular emails, a transformer-based spaCy model (`en_core_web_trf`) was used to extract organizations (ORG entities).

Observations:

* Large variety of detected organizations
* Some false positives due to ambiguity in organization names
* Most frequent organization: **"Enron"**

This suggests that the dataset likely includes emails from the well-known Enron email corpus.


## Key Learning Points

* End-to-end NLP pipeline construction
* Handling slightly imbalanced datasets
* Threshold optimization aligned with business constraints
* Topic modeling with LDA
* Topic coherence evaluation
* Word embeddings for semantic comparison
* Named Entity Recognition challenges
* Combining predictive modeling with exploratory text analytics


## Future Improvements

* Cross-validation
* Hyperparameter tuning
* More advanced classifiers (SVM, XGBoost, Transformer-based classifiers)
* Cost-sensitive learning
* Precision-Recall curve optimization
* Use of contextual embeddings (BERT) for classification


## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* spaCy
* NLTK
* Gensim
* Scikit-learn
* WordCloud


## Conclusion

This project demonstrates how a classical machine learning model (Logistic Regression + TF-IDF) can achieve excellent performance in spam detection when combined with thoughtful preprocessing and threshold tuning.

Beyond classification, the project explores the semantic structure of spam emails and applies advanced NLP techniques such as topic modeling and transformer-based NER, providing both predictive power and analytical insight.

