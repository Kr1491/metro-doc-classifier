# 🚇 Metro Document Classifier

A BERT-based document classification system for metro/railway organizations. Automatically categorizes PDF documents into six operational categories using a fine-tuned `bert-base-uncased` model, served through a Flask web application with a real-time processing queue.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🏷️ Categories

The model classifies documents into six categories:

| # | Category |
|---|----------|
| 1 | Technical & Engineering Documents |
| 2 | Passenger & Public-Facing Documents |
| 3 | Financial & Procurement Documents |
| 4 | Human Resources & Administrative Documents |
| 5 | Safety, Security & Regulatory Documents |
| 6 | Strategic & Project Management Documents |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/metro-doc-classifier.git
cd metro-doc-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

> The model is hosted on 🤗 HuggingFace Hub and is downloaded automatically on first run. No manual setup needed.

---

## 🖥️ How It Works

```
User uploads PDFs
      │
      ▼
Flask receives files → adds to thread-safe queue
      │
      ▼
Background worker picks up each file
      │
      ├─ Extracts text from page 1 via PyMuPDF (fitz)
      │
      ├─ Tokenizes with BERT tokenizer (max 512 tokens)
      │
      ├─ Runs inference → softmax → predicted class + confidence %
      │
      └─ Saves PDF into uploads/<category>/ folder
      │
      ▼
Frontend polls /file-status every 900ms → updates UI live
```

---

## 🚀 Deployment Notes

- For production, replace `app.run(debug=True)` with a WSGI server like **Gunicorn**:
  ```bash
  gunicorn -w 1 -b 0.0.0.0:8000 app:app
  ```
  Use `-w 1` (single worker) to avoid multiple instances of the background thread.

- The `file_status` dict is **in-memory only** — it resets on restart. For persistence, swap it with a SQLite DB or Redis.

---

## 📓 Training

The model was fine-tuned on a custom metro railway document dataset using `bert-base-uncased` via HuggingFace Transformers. See [`notebooks/training.ipynb`](notebooks/training.ipynb) for the full training pipeline including:

- Dataset preparation & label encoding
- Tokenization & DataLoader setup
- Fine-tuning loop with evaluation
- Saving the model in HuggingFace format (`save_pretrained`)

The trained model is hosted on HuggingFace Hub at [`YOUR_USERNAME/metro-bert-classifier`](https://huggingface.co/YOUR_USERNAME/metro-bert-classifier).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
