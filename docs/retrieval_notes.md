# Retrieval Evaluation Notes

Real top-3 results have not been recorded. This environment can reach SEC and
run Docker, but `EDGAR_IDENTITY` is not configured, so filings cannot be
downloaded through edgartools. No retrieval results are fabricated here.

Set `EDGAR_IDENTITY` to a name and contact email, start Weaviate, ingest filings,
then run each query with and without `--rerank` and fill in the result cells.
Each result should include ticker, item, the first 200 characters, and score.

| # | Query | Without rerank: top 3 | With rerank: top 3 | Status |
|---|---|---|---|---|
| 1 | What are the principal risk factors described by the company? | TODO | TODO | TODO |
| 2 | How could supply chain disruptions affect operations and costs? | TODO | TODO | TODO |
| 3 | Who are the company's main competitors and how does it differentiate? | TODO | TODO | TODO |
| 4 | What are the primary revenue drivers and how are they changing? | TODO | TODO | TODO |
| 5 | What cybersecurity threats, incidents, and controls does the company disclose? | TODO | TODO | TODO |
| 6 | What factors could reduce demand for the company's products or services? | TODO | TODO | TODO |
| 7 | What are the company's largest operating expenses and cost pressures? | TODO | TODO | TODO |
| 8 | What major capital investments or capacity expansions are planned? | TODO | TODO | TODO |
| 9 | What regulatory or legal changes could materially affect the business? | TODO | TODO | TODO |
| 10 | What risks does the company identify around international operations and currency? | TODO | TODO | TODO |