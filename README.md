# DocuMind

Chat with one or more PDFs. Answers cite the file and page they came from.

Live demo: <ADD YOUR STREAMLIT LINK>

## Why I built it
<Write 2-3 lines in your own words: which documents you wanted to query and why.>

## How it works
PDF -> text per page -> overlapping chunks (chunks never cross pages) -> Gemini embeddings
-> cosine similarity search (numpy) -> top-k chunks + recent chat turns -> Gemini answer with citations.

No vector database: for a few thousand chunks, a normalised matrix and one dot product is enough and easier to reason about.

## Evaluation
`run_eval.py` checks whether the correct page appears in the top-5 retrieved chunks for a set of
questions I wrote by hand.

| Chunk size | Questions | hit@5 |
|-----------|-----------|-------|
| 100       | <fill>    | <fill>|
| 180       | <fill>    | <fill>|
| 300       | <fill>    | <fill>|

What I learned: <write what changed and why>

## Known limits
- Scanned PDFs need OCR
- Index is in memory and rebuilt when files change
- Tables and multi-column layouts can extract badly

## Run locally
    pip install -r requirements.txt
    set GEMINI_API_KEY, then: streamlit run app.py
