
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages):
    """Split PDF pages into chunks while preserving page numbers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks
