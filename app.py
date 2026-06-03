import os
import fitz
from flask import Flask, render_template, request, jsonify
import io
import threading
import queue
import time
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

file_queue = queue.Queue()
processing_thread = None
DB_PATH = os.path.join(os.path.dirname(__file__), 'documents.db')
db_lock = threading.Lock()


def init_db():
    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    filename TEXT PRIMARY KEY,
                    category TEXT,
                    confidence REAL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    file_size INTEGER
                )
            ''')


def upsert_file_status(filename, status, category=None, confidence=None, file_size=None):
    with db_lock:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute('''
                INSERT INTO documents (filename, category, confidence, timestamp, status, file_size)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    category = excluded.category,
                    confidence = excluded.confidence,
                    timestamp = CURRENT_TIMESTAMP,
                    status = excluded.status,
                    file_size = excluded.file_size
            ''', (filename, category, confidence, status, file_size))


def get_file_status_map():
    with db_lock:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute('''
                SELECT filename, category, confidence, timestamp, status, file_size
                FROM documents
                ORDER BY timestamp DESC
            ''').fetchall()

    statuses = {}
    for filename, category, confidence, timestamp, status, file_size in rows:
        statuses[filename] = {
            'status': status,
            'category': category,
            'confidence': f"{confidence:.2f}%" if confidence is not None else None,
            'timestamp': timestamp,
            'file_size': file_size
        }
    return statuses

# --- Load the Trained BERT Model ---
MODEL_PATH = 'Kr1491/metro-bert-classifier'
tokenizer = None
model = None

print("Loading the BERT classifier...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    CATEGORIES = [
        'Technical & Engineering Documents',
        'Passenger & Public-Facing Documents', 
        'Financial & Procurement Documents',
        'Human Resources & Administrative Documents',
        'Safety, Security & Regulatory Documents',
        'Strategic & Project Management Documents'
    ]
    id_to_label = {idx: label for idx, label in enumerate(CATEGORIES)}
    
    print("BERT model loaded successfully!")
except Exception as e:
    print(f"Error loading the BERT model: {e}")
    model = None
    
# This function is now updated to return both category and confidence
def categorize_document_with_bert(text_content):
    if not model:
        raise RuntimeError("BERT model is not loaded. Cannot categorize.")
    
    inputs = tokenizer(text_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()
        
        predicted_idx = torch.argmax(logits, dim=1).item()
        confidence = float(probabilities[predicted_idx].item()) * 100
    
    return id_to_label[predicted_idx], confidence

def process_file_queue():
    while True:
        filename = None
        file_size = None
        try:
            filename, file_data = file_queue.get()
            file_size = len(file_data)
            
            upsert_file_status(filename, 'Processing', file_size=file_size)
            print(f"Processing file: {filename}")
            
            doc = fitz.open(stream=file_data, filetype="pdf")
            extracted_text = doc.get_page_text(0)
            doc.close()
            
            # The model now returns both category and confidence
            category, confidence = categorize_document_with_bert(extracted_text)

            category_folder = os.path.join(app.config['UPLOAD_FOLDER'], category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            final_file_path = os.path.join(category_folder, filename)
            
            with open(final_file_path, 'wb') as f:
                f.write(file_data)
            
            # Update the status with both category and confidence
            upsert_file_status(
                filename,
                'Completed',
                category=category,
                confidence=confidence,
                file_size=file_size
            )
            
            print(f"Successfully processed and categorized: {filename} -> {category} (Confidence: {confidence:.2f}%)")
            file_queue.task_done()
            time.sleep(1)

        except Exception as e:
            if filename:
                upsert_file_status(filename, 'Error', file_size=file_size)
            print(f"Error processing file: {filename}, Error: {e}")
            file_queue.task_done()

init_db()
processing_thread = threading.Thread(target=process_file_queue, daemon=True)
processing_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-multiple-pdfs', methods=['POST'])
def upload_multiple_pdfs():
    if not model:
        return jsonify({"error": "BERT model not loaded. Please check terminal for errors."}), 500

    if 'pdfFiles' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('pdfFiles')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    for file in files:
        if file.filename:
            file_data = file.read()
            upsert_file_status(file.filename, 'Queued', file_size=len(file_data))
            file_queue.put((file.filename, file_data))
    
    return jsonify({"message": f"{len(files)} file(s) added to the queue for processing."}), 200

@app.route('/file-status')
def get_file_status():
    return jsonify(get_file_status_map())

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5050'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host=host, port=port, debug=debug)