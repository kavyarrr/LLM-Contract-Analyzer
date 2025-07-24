import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# === Folder paths ===
DATA_DIR = "../data/"
OUTPUT_TEXT_DIR = "../outputs/extracted_texts/"
VECTOR_STORE_DIR = "../outputs/vector_store/"

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# === Step 1: Extract text from each PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

print("📄 Extracting text from PDFs...")

all_texts = {}
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, filename)
        text = extract_text_from_pdf(pdf_path)
        
        txt_file = os.path.join(OUTPUT_TEXT_DIR, filename.replace(".pdf", ".txt"))
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved text from {filename}")
        
        all_texts[filename] = text

# === Step 2: Chunk the text ===
def chunk_text(text, max_words=120):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) <= max_words:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

print("\n✂️ Chunking and embedding text...")

model = SentenceTransformer("all-MiniLM-L6-v2")
all_chunks = []
chunk_meta = []

for doc_name, text in all_texts.items():
    chunks = chunk_text(text)
    all_chunks.extend(chunks)
    chunk_meta.extend([{"doc": doc_name, "chunk_index": i} for i in range(len(chunks))])

print(f"🔹 Total chunks: {len(all_chunks)}")

# === Step 3: Create embeddings ===
embeddings = model.encode(all_chunks)

# === Step 4: Store in FAISS index ===
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))

with open(os.path.join(VECTOR_STORE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

with open(os.path.join(VECTOR_STORE_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(chunk_meta, f)

print("✅ Embeddings and metadata saved!")
