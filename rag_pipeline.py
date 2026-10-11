
from vector_store import search_vector_store
from llm_utils import generate_answer


def answer_question(index, chunks, question):
    relevant_chunks = search_vector_store(
        index=index,
        chunks=chunks,
        question=question,
        top_k=5
    )

    if not relevant_chunks:
        return {
            "answer": "No relevant sections were found in the uploaded paper.",
            "sources": []
        }

    answer = generate_answer(
        question=question,
        retrieved_chunks=relevant_chunks
    )

    sources = []
    seen_pages = set()

    for chunk in relevant_chunks:
        page = chunk["page"]

        if page not in seen_pages:
            sources.append({
                "page": page,
                "text": chunk["text"]
            })
            seen_pages.add(page)

    return {
        "answer": answer,
        "sources": sources
    }
