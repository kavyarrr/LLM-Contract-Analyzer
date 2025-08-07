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
        # Clean up the text
        page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
        page_text = page_text.strip()
        if page_text:  # Only add non-empty pages
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

# === Step 2: Improved Semantic chunking ===
def find_semantic_boundaries(text):
    """Find natural boundaries like section headers, clause numbers, etc."""
    # Common legal document patterns
    patterns = [
        r'\b(?:SECTION|Section|section)\s+[A-Z]?[0-9]*[\.\)]?\s*[A-Z]',  # Section headers
        r'\b(?:CLAUSE|Clause|clause)\s+[0-9]+[\.\)]?\s*[A-Z]',  # Clause headers
        r'\b(?:DEFINITION|Definition|definition)\s+[0-9]+[\.\)]?\s*[A-Z]',  # Definition headers
        r'\b(?:EXCLUSION|Exclusion|exclusion)\s+[0-9]+[\.\)]?\s*[A-Z]',  # Exclusion headers
        r'\b[A-Z][A-Z\s]{3,}:\s*[A-Z]',  # All caps headers
        r'\b[0-9]+[\.\)]\s*[A-Z][a-z]',  # Numbered items
    ]
    
    boundaries = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            boundaries.append(match.start())
    
    return sorted(boundaries)

def improved_semantic_chunk_pdf_text(text, filename, page_number=None, min_tokens=150, max_tokens=300, overlap_tokens=50, tokenizer=None):
    """Improved chunking with semantic boundaries and better text processing."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Find semantic boundaries
    boundaries = find_semantic_boundaries(text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    meta = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0
    i = 0
    
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(tokenizer.tokenize(sentence))

        # Handle very long sentences
        if sentence_tokens > max_tokens:
            print(f"✂️ Truncating long sentence on page {page_number} of {filename}")
            tokens = tokenizer.tokenize(sentence)[:max_tokens]
            sentence = tokenizer.convert_tokens_to_string(tokens)
            sentence_tokens = len(tokens)

        # Check if adding this sentence would exceed max_tokens
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        else:
            # If we have enough content, create a chunk
            if current_tokens >= min_tokens:
                chunk_text = " ".join(current_chunk).strip()
                chunks.append(chunk_text)
                meta.append({
                    "doc": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": current_tokens
                })
                chunk_index += 1

                # Create overlap: keep last sentences that fit within overlap_tokens
                overlap = []
                overlap_tokens_used = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(tokenizer.tokenize(sent))
                    if overlap_tokens_used + sent_tokens <= overlap_tokens:
                        overlap.insert(0, sent)
                        overlap_tokens_used += sent_tokens
                    else:
                        break
                
                current_chunk = overlap
                current_tokens = overlap_tokens_used
            else:
                # Force add the sentence if we're below min_tokens
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                i += 1

    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        chunks.append(chunk_text)
        meta.append({
            "doc": filename,
            "page": page_number,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": current_tokens
        })

    return chunks, meta

print("\n✂️ Improved chunking and embedding text...")

# Use a better sentence transformer model
model = SentenceTransformer("all-mpnet-base-v2")  # Better than all-MiniLM-L6-v2
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
all_chunks = []
chunk_meta = []

for doc_name, page_texts in all_texts.items():
    print(f"📄 Processing {doc_name}...")
    for page_text, page_num in page_texts:
        chunks, meta = improved_semantic_chunk_pdf_text(
            page_text,
            filename=doc_name,
            page_number=page_num,
            min_tokens=150,    # Increased from 100
            max_tokens=300,    # Reduced from 480 for more precise chunks
            overlap_tokens=50,  # Increased from 20 for better context
            tokenizer=tokenizer
        )
        all_chunks.extend(chunks)
        chunk_meta.extend(meta)

print(f"🔹 Total chunks: {len(all_chunks)}")
print(f"🔹 Average chunk length: {np.mean([len(chunk.split()) for chunk in all_chunks]):.1f} words")

# === Step 3: Create embeddings ===
print("🔍 Creating embeddings...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

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

print("✅ Improved embeddings and metadata saved!")
print(f"📊 Index contains {len(all_chunks)} chunks with {dimension}-dimensional embeddings")
