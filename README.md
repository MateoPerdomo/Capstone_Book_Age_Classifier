# 📚 Book Age Classifier

> A multi-feature machine learning system that automatically classifies book content by age suitability, built for parents, educators, and publishers.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/SVM%20Accuracy-75.1%25-brightgreen?style=flat-square)
![IE University](https://img.shields.io/badge/IE%20University-Capstone%20Project-red?style=flat-square)

---

## Overview

Children and teenagers have greater access than ever to books, e-readers, and online platforms, yet age labelling still relies heavily on expert judgment, which is slow and inconsistent at scale. **Book Age Classifier** addresses this gap with a transparent, interpretable machine learning pipeline that assigns one of four age suitability labels to any book text excerpt:

| Label | Age Range | Description |
|-------|-----------|-------------|
| **+4** | Early readers (4 to 8) | Simple vocabulary, short sentences, positive emotional tone |
| **+10** | Middle grade (8 to 12) | Moderate complexity, adventure and mystery themes |
| **+12** | Young adult (12 to 18) | Identity themes, mild conflict, darker emotional content |
| **+18** | Adult | Complex language, mature themes, sensitive content |

Rather than using a black-box deep learning model, this system extracts **23 human-interpretable features** across five analytical domains and explains *why* each classification was made, making it genuinely useful to non-technical users.

---

## Features

The classifier operates on 23 linguistically motivated features across five domains:

| Domain | Features | Example Signal |
|--------|----------|----------------|
| **Readability** | Flesch Reading Ease, FK Grade, Gunning Fog, Dale-Chall, SMOG | +4 texts average a Reading Ease of 88.1 vs 51.6 for +18 |
| **Linguistic** | Avg sentence length, vocab richness, word length, long word ratio, content word ratio | Sentence length doubles from 13.4 words (+4) to 31.4 words (+18) |
| **Sensitivity** | Violence score, profanity score, adult score, drug score | Adult score increases 185x from +4 to +18 |
| **Sentiment** | VADER positive, negative, neutral, compound | Compound sentiment drops from 0.753 (+4) to 0.008 (+18) |
| **Style** | Dialogue ratio, exclamation ratio, question ratio | Dialogue ratio is 8.5x higher in +10 than in +18 texts |

---

## Model Performance

Five classical machine learning classifiers were evaluated on a balanced held-out test set of 1,600 samples (400 per class):

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Logistic Regression | 0.744 ± 0.006 | 72.8% |
| Random Forest | 0.739 ± 0.011 | 74.9% |
| Gradient Boosting | 0.748 ± 0.012 | 74.2% |
| **SVM (RBF) ✓** | **0.754 ± 0.009** | **75.1%** |
| Naive Bayes | 0.642 ± 0.032 | 64.1% |

The SVM outperforms a single-feature baseline (word count only) by **+31.4 percentage points**, confirming the value of the multi-dimensional feature engineering approach.

Per-class F1 scores for the best model:

| Age Group | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| +4 | ~0.87 | ~0.92 | **~0.89** |
| +10 | ~0.68 | ~0.70 | ~0.69 |
| +12 | ~0.65 | ~0.63 | ~0.64 |
| +18 | ~0.81 | ~0.77 | ~0.79 |

The +4 and +18 categories are classified with high confidence. The +10/+12 boundary is the hardest to classify, which is expected given that young adult literature sits at the transition between childhood and adulthood, a finding consistent with the existing literature.

---

## Project Structure

```
book-age-classifier/
│
├── app/
│   └── 06_app_2.py              # Desktop GUI application (tkinter)
│
├── data/
│   └── data_raw.csv             # Balanced dataset (8,000 samples, 4 classes)
│
├── models/
│   ├── best_model.pkl           # Trained SVM classifier
│   ├── scaler.pkl               # Fitted StandardScaler
│   ├── label_encoder.pkl        # LabelEncoder for age groups
│   └── feature_cols.pkl         # List of 23 feature column names
│
├── notebooks/
│   └── main_notebook.ipynb      # Full pipeline: data, features, EDA, modelling
│
├── report/
│   └── capstone_report.pdf      # Full academic report (45 pages)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/MateoPerdomo/book-age-classifier.git
cd book-age-classifier
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download NLTK data (required for feature extraction):**
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

---

## How to Run

### Option A: Desktop Application

The desktop app lets any user paste a book excerpt and receive an instant age classification, a confidence breakdown, and a plain-language explanation of the top features driving the result.

Make sure the four model artefacts are in your working directory, then run:

```bash
python app/06_app_2.py
```

The app will open a GUI window. Paste any book excerpt (minimum 10 words), click **Classify**, and the results will appear instantly.

### Option B: Jupyter Notebook

To explore the full pipeline including data collection, feature engineering, EDA, modelling, and evaluation, open the notebook:

```bash
jupyter notebook notebooks/main_notebook.ipynb
```

The notebook is self-contained and runs top-to-bottom. Note that the data collection cells call the HuggingFace API and require an internet connection.

---

## Example Predictions

| Excerpt Type | Predicted Label | Confidence | Key Features |
|---|---|---|---|
| Rabbit learning to share (children's story) | **+4** | 89.2% | Flesch Ease: 94.7, avg sentence: 8.2 words, compound sentiment: +0.87 |
| Kid solving a school mystery | **+10** | 71.4% | FK Grade: 7.8, dialogue ratio: 0.19, compound sentiment: +0.66 |
| Crime fiction, detective at a murder scene | **+18** | 73.0% | Compound sentiment: -0.96, violence score: 25.3, avg sentence: 24.1 words |

---

## Explainability

A core design principle of this project is **transparency**. Instead of returning a black-box verdict, the application shows the five features that most influenced the classification, with directional indicators:

```
sentiment_negative  ▲ toward +18   contribution=+1.09   raw=0.112
violence_score      ▲ toward +18   contribution=+0.80   raw=10.6
num_words           ▼ away from +18  contribution=-0.68  raw=138.0
```

This means a parent or teacher can understand not just *what* the model decided, but *why*, directly addressing the black-box problem identified in the literature.

---

## Data Sources

| Source | Age Group | Type |
|--------|-----------|------|
| [Eitlani/goodreads](https://huggingface.co/datasets/Eitlani/goodreads) | All | Real book descriptions |
| [pszemraj/goodreads-bookgenres](https://huggingface.co/datasets/pszemraj/goodreads-bookgenres) | All | Real book descriptions with genre tags |
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | +4 top-up | Synthetic children's stories |
| [ajibawa-2023/Children-Stories-Collection](https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection) | +10 top-up | Synthetic middle-grade stories |

All texts were truncated to 200 words and filtered to a minimum of 20 words. The final dataset contains exactly 2,000 samples per class (8,000 total).

---

## Limitations

Being transparent about limitations is as important as reporting results:

- **75.1% accuracy** falls short of the 80% target, primarily due to overlap between the +10 and +12 categories
- The model operates on **book descriptions**, not full texts. Summaries are written to be engaging and may not fully reflect the book's linguistic register
- **Keyword-based sensitivity detection** does not consider context, meaning benign uses of sensitive words (e.g. "smoke" from a chimney) can inflate sensitivity scores
- The feature set and keyword lists are **English-only** and reflect Anglophone cultural norms
- Labels are derived from **crowd-sourced Goodreads shelves**, not expert annotation

---

## Ethical Considerations

- **Cultural bias:** keyword lists are grounded in Western, Anglophone contexts and may not generalise to multicultural settings
- **Over-restriction risk:** a conservative classifier may flag age-appropriate material as unsuitable. The system is designed as a recommendation tool, not a definitive gatekeeper
- **Human oversight:** no automated system should be the sole arbiter of what is appropriate for minors. Outputs should always be reviewed by a human
- **Privacy:** the desktop application is fully client-side and does not transmit any text to a server

---

## Future Work

- Integrating contextual embeddings from transformer models (BERT, RoBERTa) to better capture thematic nuance, especially at the +10/+12 boundary
- Extending the pipeline to classify full book chapters rather than short excerpts
- Collecting a human-annotated ground-truth dataset across genres, languages, and cultures
- Adding multi-language support with language-specific readability formulas and keyword sets
- Conducting a user study with parents, teachers, and librarians to validate the explainability mechanism

---

## Academic Context

This project was completed as a Bachelor's capstone at **IE University, School of Science & Technology**, under the supervision of **Suzan Awinat** (Universidad Autónoma de Madrid / SciTech Labs, IE University).

**Degree:** Bachelor of Data and Business Analytics  
**Author:** Mateo Perdomo Andrés

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*If you find this project useful or interesting, feel free to star the repository ⭐*
