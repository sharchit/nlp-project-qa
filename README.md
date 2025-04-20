# Abstractive Question Answering using Transformers on IITP Student Life Dataset

This project demonstrates training and evaluating three state-of-the-art transformer models— BART, Pegasus, and T5 for abstractive question answering. The models are trained on a dataset of question-context-answer triplets generated for self created IITP student life dataset  and evaluated using ROUGE and BERTScore metrics.

---

## Project Structure

- `bart_qa.py` – Training and inference using the BART model.
- `pegasus_qa.py` – Training and inference using the Pegasus model.
- `t5_qa.py` – Training and inference using the T5 model.
- `final_dataset.csv` – Dataset containing 1,612 rows of context, question, and answer pairs used for training.
- `bart_results.json` – Evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore precision/recall/F1) for BART.
- `pegasus_results.json` – Evaluation metrics for Pegasus.
- `t5_results.json` – Evaluation metrics for T5.

---

## Running Code 

### 1. Install Dependencies

```bash     
pip install -r requirements.txt
```

### Contributions
- AS. Poornash (2101CS01):
    - Dataset Creation (~35%)
    - Fine-tuning and evaluating the models.
- Archit Sharma (2101AI05):
    - Dataset Creation (~30%)
    - Model training and interface for demonstration
- Aman Verma (2101AI37):
    - Dataset creation (~35%) and verification.
    - Analyzed results and prepared reports.
