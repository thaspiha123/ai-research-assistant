"""
citations.py
------------
Builds APA 7th edition and IEEE references (and in-text citations) for the
research papers uploaded to the assistant.

No external libraries are needed, so this file cannot fail because of a
missing package.

Text wrapped in *asterisks* is meant to be shown in italics (Markdown).
Use plain() to remove the asterisks for copy-paste into Word.
"""

import re
from typing import Dict, List, Optional, Tuple

PAPER_KINDS = ["Journal article", "Conference paper", "Preprint / arXiv / report"]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _clean(value) -> str:
    return (value or "").strip() if isinstance(value, str) else ""


def _end(sentence: str) -> str:
    """Make sure a sentence ends with exactly one full stop (or ? !)."""
    sentence = sentence.strip()
    if not sentence:
        return ""
    return sentence if sentence[-1] in ".?!" else sentence + "."


def plain(text: str) -> str:
    """Remove the Markdown italics markers."""
    return text.replace("*", "")


def _pages_norm(pages: str) -> str:
    """'45-67' or '45--67' -> '45–67' (en dash)."""
    return re.sub(r"\s*[-–—]+\s*", "–", _clean(pages))


def _pp_label(pages: str) -> str:
    pages = _pages_norm(pages)
    if not pages:
        return ""
    return ("pp. " if "–" in pages else "p. ") + pages


def _doi_url(doi: str) -> str:
    doi = _clean(doi)
    if not doi:
        return ""
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return "https://doi.org/" + doi


def _doi_plain(doi: str) -> str:
    return _doi_url(doi).replace("https://doi.org/", "")


# ----------------------------------------------------------------------------
# Author handling
# ----------------------------------------------------------------------------
def parse_authors(raw: str) -> List[Tuple[str, str]]:
    """
    Turn the text typed by the user into [(last, first), ...].

    Accepted: one author per line, or separated by ';' or 'and' or '&'.
    Each author can be 'First Last' or 'Last, First'.
    """
    raw = _clean(raw).replace("\r", "")
    if not raw:
        return []
    parts = re.split(r"[;\n]+|\s+&\s+|\s+and\s+", raw, flags=re.I)
    authors = []
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        if "," in part:
            last, first = [p.strip() for p in part.split(",", 1)]
        else:
            tokens = part.split()
            if len(tokens) == 1:
                last, first = tokens[0], ""
            else:
                last, first = tokens[-1], " ".join(tokens[:-1])
        authors.append((last, first))
    return authors


def _initials(first: str) -> List[str]:
    tokens = [t for t in re.split(r"[\s.\-]+", first.strip()) if t]
    return [t[0].upper() + "." for t in tokens]


def _apa_name(last: str, first: str) -> str:
    ini = " ".join(_initials(first))
    return f"{last}, {ini}" if ini else last


def _ieee_name(last: str, first: str) -> str:
    ini = " ".join(_initials(first))
    return f"{ini} {last}" if ini else last


def _apa_author_list(authors: List[Tuple[str, str]]) -> str:
    names = [_apa_name(l, f) for l, f in authors]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, & {names[1]}"
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", . . . " + names[-1]


def _ieee_author_list(authors: List[Tuple[str, str]]) -> str:
    names = [_ieee_name(l, f) for l, f in authors]
    if not names:
        return ""
    if len(names) > 6:
        return names[0] + " et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


# ----------------------------------------------------------------------------
# APA 7th edition
# ----------------------------------------------------------------------------
def apa_reference(meta: Dict[str, str]) -> str:
    authors = parse_authors(meta.get("authors", ""))
    year = _clean(meta.get("year")) or "n.d."
    title = _clean(meta.get("title")) or "Untitled paper"
    kind = meta.get("kind") or PAPER_KINDS[0]
    venue = _clean(meta.get("venue"))
    volume = _clean(meta.get("volume"))
    issue = _clean(meta.get("issue"))
    pages = _pages_norm(meta.get("pages", ""))
    link = _doi_url(meta.get("doi", "")) or _clean(meta.get("url"))

    preprint = kind == PAPER_KINDS[2]
    author_text = _apa_author_list(authors)
    title_text = f"*{_end(title)}*" if preprint else _end(title)

    if author_text:
        head = f"{author_text} ({year}). "
        body = title_text + " "
    else:  # APA: title moves to the author position
        head = f"{title_text} ({year}). "
        body = ""

    if kind == PAPER_KINDS[0]:
        source = f"*{venue}*" if venue else ""
        if volume:
            source += f", *{volume}*" if source else f"*{volume}*"
            if issue:
                source += f"({issue})"
        if pages:
            source += f", {pages}" if source else pages
        source = _end(source) if source else ""
    elif kind == PAPER_KINDS[1]:
        source = f"In *{venue}*" if venue else ""
        if pages:
            source += f" (pp. {pages})" if source else f"(pp. {pages})"
        source = _end(source) if source else ""
    else:
        source = _end(venue) if venue else ""

    result = head + body + (source + " " if source else "") + link
    return re.sub(r"\s+", " ", result).strip()


def apa_intext(meta: Dict[str, str], page: Optional[int] = None) -> str:
    authors = parse_authors(meta.get("authors", ""))
    year = _clean(meta.get("year")) or "n.d."
    if not authors:
        short = " ".join(_clean(meta.get("title")).split()[:3]) or "Untitled"
        who = f'"{short}"'
    elif len(authors) == 1:
        who = authors[0][0]
    elif len(authors) == 2:
        who = f"{authors[0][0]} & {authors[1][0]}"
    else:
        who = f"{authors[0][0]} et al."
    suffix = f", p. {page}" if page else ""
    return f"({who}, {year}{suffix})"


# ----------------------------------------------------------------------------
# IEEE
# ----------------------------------------------------------------------------
def ieee_reference(meta: Dict[str, str], number: int = 1) -> str:
    authors = parse_authors(meta.get("authors", ""))
    year = _clean(meta.get("year"))
    title = _clean(meta.get("title")) or "Untitled paper"
    kind = meta.get("kind") or PAPER_KINDS[0]
    venue = _clean(meta.get("venue"))
    volume = _clean(meta.get("volume"))
    issue = _clean(meta.get("issue"))
    pp = _pp_label(meta.get("pages", ""))
    doi = _doi_plain(meta.get("doi", ""))
    url = _clean(meta.get("url"))

    author_text = _ieee_author_list(authors)
    start = f'{author_text}, "{title},"' if author_text else f'"{title},"'

    parts: List[str] = []
    if kind == PAPER_KINDS[0]:
        if venue:
            parts.append(f"*{venue}*")
        if volume:
            parts.append(f"vol. {volume}")
        if issue:
            parts.append(f"no. {issue}")
        if pp:
            parts.append(pp)
        if year:
            parts.append(year)
    elif kind == PAPER_KINDS[1]:
        if venue:
            parts.append(f"in *{venue}*")
        if year:
            parts.append(year)
        if pp:
            parts.append(pp)
    else:
        if venue:
            parts.append(f"*{venue}*")
        if year:
            parts.append(year)

    if parts:
        text = start + " " + ", ".join(parts)
    else:
        text = start[:-2] + '."' if start.endswith(',"') else start
    text = text.rstrip(",")
    if doi:
        text += f", doi: {doi}."
    else:
        text = _end(text)
        if url:
            text += f" [Online]. Available: {url}"
    return re.sub(r"\s+", " ", f"[{number}] {text}").strip()


def ieee_intext(number: int, page: Optional[int] = None) -> str:
    return f"[{number}, p. {page}]" if page else f"[{number}]"


# ----------------------------------------------------------------------------
# Self-test:  python citations.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demo = {
        "title": "Retrieval-augmented generation for knowledge-intensive NLP tasks",
        "authors": "Patrick Lewis; Ethan Perez; Aleksandra Piktus",
        "year": "2020",
        "kind": PAPER_KINDS[1],
        "venue": "Advances in Neural Information Processing Systems",
        "volume": "", "issue": "", "pages": "9459-9474", "doi": "", "url": "",
    }
    print("APA :", apa_reference(demo))
    print("IEEE:", ieee_reference(demo, 1))
    print("In-text APA :", apa_intext(demo, 3))
    print("In-text IEEE:", ieee_intext(1, 3))
