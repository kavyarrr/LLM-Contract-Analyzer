import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import re
from transformers import AutoTokenizer

# === Folder paths ===
DATA_DIR = "../data/"
OUTPUT_TEXT_DIR = "../outputs/extracted_texts/"
VECTOR_STORE_DIR = "../outputs/vector_store/"

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# === Step 1: Extract text from each PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        texts.append((page_text, page_num))
    return texts  # List of (text, page_number)

print("📄 Extracting text from PDFs...")

all_texts = {}
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, filename)
        page_texts = extract_text_from_pdf(pdf_path)
        
        txt_file = os.path.join(OUTPUT_TEXT_DIR, filename.replace(".pdf", ".txt"))
        with open(txt_file, "w", encoding="utf-8") as f:
            for page_text, page_num in page_texts:
                f.write(f"\n--- Page {page_num} ---\n")
                f.write(page_text)
        print(f"✅ Saved text from {filename}")
        
        all_texts[filename] = page_texts  # Store list of (text, page_number)

# === Step 2: Semantic chunking ===
def semantic_chunk_pdf_text(text, filename, page_number=None, min_tokens=100, max_tokens=150, overlap_tokens=20, tokenizer=None):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    chunks = []
    meta = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(tokenizer.tokenize(sentence))

        # 🆕 Truncate long sentences that exceed max_tokens
        if sentence_tokens > max_tokens:
            print(f"✂️ Truncating long sentence on page {page_number} of {filename}")
            tokens = tokenizer.tokenize(sentence)[:max_tokens]
            sentence = tokenizer.convert_tokens_to_string(tokens)
            sentence_tokens = len(tokens)

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        else:
            if current_tokens < min_tokens and i < len(sentences):
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                i += 1
            else:
                chunk_text = " ".join(current_chunk).strip()
                chunks.append(chunk_text)
                meta.append({
                    "doc": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text 
                })
                chunk_index += 1

                # Overlap: keep last overlap_tokens from current_chunk
                if overlap_tokens > 0 and current_chunk:
                    overlap = []
                    tokens_so_far = 0
                    for sent in reversed(current_chunk):
                        sent_tokens = len(tokenizer.tokenize(sent))
                        if tokens_so_far + sent_tokens <= overlap_tokens:
                            overlap.insert(0, sent)
                            tokens_so_far += sent_tokens
                        else:
                            break
                    current_chunk = overlap
                    current_tokens = sum(len(tokenizer.tokenize(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0

    # Add final leftover chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        chunks.append(chunk_text)
        meta.append({
            "doc": filename,
            "page": page_number,
            "chunk_index": chunk_index,
            "text": chunk_text
        })

    return chunks, meta


print("\n✂️ Chunking and embedding text...")

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
all_chunks = []
chunk_meta = []

for doc_name, page_texts in all_texts.items():
    for page_text, page_num in page_texts:
        chunks, meta = semantic_chunk_pdf_text(
            page_text,
            filename=doc_name,
            page_number=page_num,
            min_tokens=100,
            max_tokens=480,  
            overlap_tokens=20,
            tokenizer=tokenizer
        )
        all_chunks.extend(chunks)
        chunk_meta.extend(meta)

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
