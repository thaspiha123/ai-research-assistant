
from llm_utils import generate_answer


def _generate_from_pages(pages, task, max_chars=24000):
    """Generate a feature response from page-aware PDF text."""
    if not pages:
        return "No readable content was found in the uploaded paper."

    sections = []
    remaining = max_chars

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text or remaining <= 0:
            continue

        page_text = page_text[:remaining]
        sections.append(f"[Page {page['page']}]\n{page_text}")
        remaining -= len(page_text)

    if not sections:
        return "No readable content was found in the uploaded paper."

    context = "\n\n".join(sections)

    # Reuse the working Gemini integration.
    # Pass one context chunk so the model sees page references.
    return generate_answer(
        question=(
            f"{task}\n\n"
            "Use the supplied paper text as evidence. "
            "Cite page numbers as [Page N]. "
            "Do not invent results, references, or research findings. "
            "If evidence is insufficient, state that clearly."
        ),
        retrieved_chunks=[{
            "page": "multiple pages",
            "text": context
        }]
    )


def generate_summary(pages):
    return _generate_from_pages(
        pages,
        """
Create a structured research paper summary with:
1. Research title (if available)
2. Abstract or overview
3. Problem statement
4. Objectives
5. Methodology
6. Main findings and results
7. Conclusion
8. Limitations and future scope

Use clear, concise language. Cite page numbers for claims.
Do not invent missing information.
"""
    )


def generate_smart_notes(pages):
    return _generate_from_pages(
        pages,
        """
Generate student-friendly smart notes using these headings:
1. Topic and overview
2. Objectives
3. Important concepts and definitions
4. Methodology
5. Key findings
6. Applications
7. Limitations and future scope
8. Quick revision points

Use simple English, bullet points, and bold important terms.
Add page citations for factual claims. Do not invent information.
"""
    )


def explain_paper_simply(pages):
    return _generate_from_pages(
        pages,
        """
Explain this research paper in simple English for a college student.
Explain the problem, main idea, methodology, results, and importance.
Define difficult technical terms and give simple examples when useful.
Preserve the meaning of the paper and cite page numbers.
"""
    )


def generate_research_gaps(pages):
    return _generate_from_pages(
        pages,
        """
Identify potential research gaps based on the provided paper.
Separate:
1. Limitations explicitly stated by the authors
2. Future work explicitly suggested by the authors
3. Potential gaps inferred from the paper, with reasoning
4. Possible future research questions

Cite relevant pages. Clearly label inferred gaps as suggestions.
Do not claim that a topic has never been researched unless the
provided evidence supports that claim.
"""
    )


def generate_quiz(pages, number=10):
    number = max(1, min(int(number), 20))

    return _generate_from_pages(
        pages,
        f"""
Create {number} quiz questions based on the research paper.
Include a mix of multiple-choice questions and short-answer questions.
For multiple-choice questions, provide four options and mark the
correct answer. Provide a brief explanation and page citation.
Include an answer key at the end. Do not invent facts.
"""
    )


def generate_interactive_quiz(pages, number=5):
    number = max(1, min(int(number), 10))

    return _generate_from_pages(
        pages,
        f"""
Create {number} multiple-choice questions to test a student's
understanding of this research paper.
Each question must have four options labelled A, B, C, and D.
Put all correct answers and brief explanations in a separate
answer key at the end. Cite the supporting page for each answer.
Do not reveal the answer immediately after each question.
"""
    )


def generate_research_mentor_response(pages, question):
    return _generate_from_pages(
        pages,
        f"""
Act as a friendly AI Research Mentor for a college student.
Answer the student's request using the paper as context.
Give practical, step-by-step guidance when appropriate.
Distinguish facts from the paper from your own suggestions.
Cite paper-based claims and do not invent research findings.

Student's request:
{question}
"""
    )


def compare_papers(papers):
    """
    papers: list of dictionaries containing:
    {"name": filename, "pages": extracted_pages}
    """
    if len(papers) < 2:
        return "Please upload at least two research papers to compare."

    combined = []

    for paper in papers:
        name = paper.get("name", "Research paper")
        pages = paper.get("pages", [])
        sections = [f"Paper: {name}"]

        for page in pages:
            text = page.get("text", "").strip()
            if text:
                sections.append(
                    f"[Page {page['page']}]\n{text[:3000]}"
                )

        combined.append("\n".join(sections))

    context = "\n\n--- PAPER SEPARATOR ---\n\n".join(combined)

    return generate_answer(
        question=(
            "Compare these research papers in a structured format. "
            "Discuss objectives, methods, datasets (if stated), "
            "findings, limitations, similarities, and differences. "
            "Identify potential research gaps supported by the papers. "
            "Cite the paper name and page number for claims. "
            "Do not assume missing details."
        ),
        retrieved_chunks=[{
            "page": "multiple pages",
            "text": context
        }]
    )

from vector_store import search_vector_store


def find_citations(index, chunks, question, top_k=5):
    """
    Find relevant passages from the uploaded research paper
    and return their original PDF page numbers.
    """
    if index is None or not chunks:
        return []

    if not question or not question.strip():
        return []

    relevant_chunks = search_vector_store(
        index=index,
        chunks=chunks,
        question=question.strip(),
        top_k=top_k
    )

    citations = []
    seen = set()

    for chunk in relevant_chunks:
        page = chunk.get("page")
        text = chunk.get("text", "").strip()

        if not text:
            continue

        # Avoid showing duplicate passages from the same page.
        key = (page, text)

        if key in seen:
            continue

        seen.add(key)

        citations.append({
            "page": page,
            "text": text
        })

    return citations
