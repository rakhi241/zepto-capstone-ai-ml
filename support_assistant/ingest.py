from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="zepto_support"
)

all_chunks = []
all_ids = []
all_metadatas = []

for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
    text = doc_path.read_text(encoding="utf-8").strip()

    words = text.split()

    for i in range(0, len(words), 120):
        chunk = " ".join(words[i:i + 120])

        if chunk:
            chunk_id = f"{doc_path.stem}_chunk_{i // 120}"

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "document_id": doc_path.stem,
                "chunk_id": chunk_id
            })

print(f"Loaded {len(all_chunks)} chunks.")

print("Creating embeddings...")
embeddings = model.encode(
    all_chunks,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas
)

print("Task 1 completed successfully!")
print(f"Stored {collection.count()} chunks in ChromaDB.")