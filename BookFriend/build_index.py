import os, re, faiss, pickle
from sentence_transformers import SentenceTransformer

CHAPTERS_FOLDER = "chapters"
INDEX_PATH = "semantic_index.faiss"
MAPPING_PATH = "semantic_mapping.pkl"
CHUNK_SIZE = 800
OVERLAP = 100


def smart_chunking(text, size=800, overlap=100):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf, n = [], [], 0

    for s in sents:
        if n + len(s) > size and buf:
            chunks.append(" ".join(buf))
            buf = [buf[-1], s] if overlap else [s]
            n = len(" ".join(buf))
        else:
            buf.append(s)
            n += len(s)

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def build_index():
    if not os.path.exists(CHAPTERS_FOLDER):
        print(f"❌ Error: '{CHAPTERS_FOLDER}' folder not found.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts, mapping = [], []

    for fname in sorted(os.listdir(CHAPTERS_FOLDER)):
        if not fname.endswith(".txt"):
            continue

        content = open(os.path.join(CHAPTERS_FOLDER, fname), encoding="utf-8").read()
        if not content.strip():
            continue

        for chunk in smart_chunking(content, CHUNK_SIZE, OVERLAP):
            texts.append(chunk)
            mapping.append((fname, chunk))

    if not texts:
        print("❌ No text found to index!")
        return

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(mapping, open(MAPPING_PATH, "wb"))

    print(f"✅ Index built! Total chunks: {len(texts)}")


if __name__ == "__main__":
    build_index()
