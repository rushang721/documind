from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    doc: str
    page: int
    text: str


def read_pdf(file):
    """Return [(page_number, text)] for pages that actually contain text."""
    pages = []
    for number, page in enumerate(PdfReader(file).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def split_pages(pages, doc_name, size=180, overlap=30):
    """Word-based sliding window. Chunks never cross a page, so page citations stay exact."""
    step = size - overlap
    chunks = []
    for number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            part = words[start:start + size]
            if start > 0 and len(part) < 15:
                continue  # tiny tail, already covered by the overlap
            chunks.append(Chunk(doc_name, number, " ".join(part)))
    return chunks
