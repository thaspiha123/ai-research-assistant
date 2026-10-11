
import faiss
import numpy as np

from embeddings import generate_embeddings


def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks available.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


def search_vector_store(index, chunks, question, top_k=3):
    question_embedding = generate_embeddings([question])
    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    top_k = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        if idx >= 0:
            results.append(chunks[int(idx)])

    return results
