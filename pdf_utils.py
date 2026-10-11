
from pypdf import PdfReader
from io import BytesIO


def extract_text_from_pdf(pdf_file):
    """Extract text from each PDF page."""

    reader = PdfReader(BytesIO(pdf_file.getvalue()))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    if not pages:
        raise ValueError(
            "No readable text found in this PDF."
        )

    return pages
