import faiss, pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "semantic_index.faiss"
MAPPING_PATH = "semantic_mapping.pkl"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def load_semantic_index():
    return faiss.read_index(INDEX_PATH), pickle.load(open(MAPPING_PATH, "rb"))

def semantic_search(query, index, mapping, top_k=5):
    vec = MODEL.encode([query], convert_to_numpy=True)
    dists, idxs = index.search(vec, top_k)
    return [(mapping[i][0], mapping[i][1], d) for i, d in zip(idxs[0], dists[0])]
