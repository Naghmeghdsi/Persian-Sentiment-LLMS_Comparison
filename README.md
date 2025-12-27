# Persian-Sentiment-LLMS_Comparison
Code and experiments for a course project on adapting large language models to Persian sentiment analysis. We compare a prompt‑based baseline with a parameter‑efficient fine‑tuning (PEFT) approach on the ParsiAI/digikala-sentiment-analysis dataset, and analyse the trade‑offs between accuracy, training cost and number of trainable parameters.
# Persian Sentiment Analysis with LLMs: Prompt vs. PEFT

This repository contains the code and experiments for a course project on adapting large language models (LLMs) to **Persian sentiment analysis**.  
We compare a **prompt-based baseline** with a **parameter-efficient fine-tuning (PEFT)** approach on a Digikala review dataset, analysing the trade-offs between classification performance and training cost.

---

## 1. Project Overview

- **Task:** Sentence-level sentiment classification (e.g. positive vs. negative) for Persian user reviews.
- **Language:** Persian.
- **Objective:**  
  Evaluate and compare prompt-based inference and parameter-efficient fine-tuning (PEFT) methods for Persian sentiment classification in terms of performance and efficiency.

The project follows the course requirements and covers the full pipeline from dataset preparation to model adaptation, evaluation, and comparative analysis.

Evaluation is performed using standard classification metrics such as **Accuracy** and **macro F1-score**.

---

## 2. Dataset

- **Source:** Hugging Face Datasets  
  `ParsiAI/digikala-sentiment-analysis`
- **Content:** Persian product reviews from the Digikala e-commerce platform, with sentiment labels (e.g. positive / negative).
- **Access:** The dataset is loaded directly via the `datasets` library; no manual download is required.

Example loading code:

```python
from datasets import load_dataset

dataset = load_dataset("ParsiAI/digikala-sentiment-analysis")
print(dataset)
```
Train, validation, and test splits are created programmatically and used consistently across all experiments.
---
## 3. Setup & Installation

To set up the environment for this project:

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
2. Install required dependencies:

```bash

pip install -r requirements.txt
```
All experiments are designed to run on Google Colab or any similar environment with access to a GPU.
---

### 4. How to Run the Experiments
### 4.1. Exploratory Data Analysis & Data Splits
Run:

notebooks/01_eda.ipynb

This notebook:

Loads the Digikala sentiment dataset.

Performs basic exploratory analysis (class distribution, text length, etc.).

Creates stratified **train / validation / test** splits (e.g., 80/10/10).

Optionally saves processed splits under data/.


### 4.2. Prompt-Based Baseline (No Training)

Run:

notebooks/02_prompt_baseline.ipynb

This notebook:

Loads a multilingual or Persian pretrained transformer model from the Hugging Face Hub.

Designs Persian instruction / few-shot prompts for sentiment classification.

Applies the prompts to the test set to predict sentiment labels.

Computes Accuracy and macro F1-score.

This experiment serves as the zero-shot or few-shot baseline.

### 4.3. PEFT / LoRA Fine-Tuning

Run:

notebooks/03_peft_finetune.ipynb

This notebook:

Uses the same base model and tokenizer as the prompt baseline.

Configures a PEFT method (e.g., LoRA) to train only a small subset of parameters.

Fine-tunes the model on the training split and tunes hyperparameters using the validation split.

Evaluates on the test split using Accuracy and macro F1-score.

Records additional information such as the number of trainable parameters and training time when available.
---
### 5. Evaluation & Comparative Analysis

Evaluation includes:

Quantitative Metrics

Accuracy

Precision, Recall

Macro F1-score

Qualitative Analysis

Examples of correctly and incorrectly classified reviews

Error analysis (e.g., irony, mixed sentiment, noisy or informal Persian text)

Comparison Dimensions

Prompt-based vs. PEFT approaches:

Classification performance

Computational efficiency (training cost, parameter count)

Practical considerations (robustness, ease of use)

The final report and presentation summarise the task, dataset, methods, results, limitations, and possible future improvements.
---
### 6. Project Status

This project is developed as part of an academic course and is currently under active development.
