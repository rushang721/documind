"""Measure retrieval quality: does the right page show up in the top-k results?

Usage: python run_eval.py my.pdf questions.json [chunk_size]
"""
import json
import os
import sys

from embed import embed_texts
from ingest import read_pdf, split_pages
from retrieve import top_chunks


def main(pdf_path, questions_path, size=180, k=5):
    api_key = os.environ["GEMINI_API_KEY"]
    with open(pdf_path, "rb") as f:
        chunks = split_pages(read_pdf(f), os.path.basename(pdf_path), size=size)
    matrix = embed_texts([c.text for c in chunks], api_key)

    with open(questions_path) as f:
        questions = json.load(f)

    found = 0
    for item in questions:
        pages = [c.page for c, _ in top_chunks(item["q"], chunks, matrix, api_key, k)]
        ok = item["page"] in pages
        found += ok
        print("PASS" if ok else "MISS", "|", item["q"], "| expected", item["page"], "| got", pages)
    print(f"\nchunk size {size}, hit@{k}: {found}/{len(questions)} = {found / len(questions):.0%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 180)
