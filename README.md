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

