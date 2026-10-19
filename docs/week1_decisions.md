# Week 1 Decisions

- **Hybrid retrieval:** Weaviate hybrid search combines keyword and vector retrieval with `alpha=0.5`, keeping lexical matches useful alongside semantic similarity for filing language.
- **Chunking:** Filing text is divided into chunks capped at 700 tokens with up to 100 tokens of overlap. The overlap retains nearby context across boundaries; sentence and paragraph boundaries are preferred when splitting.
- **Citation verification:** A citation is marked verified when its quote, after case-folding and whitespace normalization, is a substring of the referenced retrieved chunk. This is a lightweight exact-text check, not semantic validation.
- **Known limitations:** Ingest job state is held in process memory and is lost on restart; the API has no authentication; citation verification does not establish that a quote supports the answer. The integration tests require a running Weaviate instance.