# finsight-agents

## 5-minute quickstart

### 1. Clone and configure the environment

```bash
git clone https://github.com/SerpilUmayGedikli/finsight-agents.git
cd finsight-agents
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` and `EDGAR_IDENTITY` before ingesting and asking questions. The API defaults to the Anthropic provider.

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
EMBEDDER=local
RETRIEVAL_K=5
RERANK_DEFAULT=false
LOG_LEVEL=INFO
EDGAR_IDENTITY="Your Name your.email@example.com"
```

### 2. Start the API stack

```bash
docker compose up --build -d
```

Check the app is alive:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 3. Ingest AAPL and MSFT 10-K data

```bash
curl -sS -X POST http://localhost:8000/ingest \
	-H "Content-Type: application/json" \
	-d '{"tickers":["AAPL","MSFT"],"form_type":"10-K","limit":2}'
```

This returns a job ID. Poll the status endpoint until it finishes:

```bash
curl http://localhost:8000/ingest/<job_id>
```

When the status is `done`, the vector store is ready for questions.

### 4. Ask a question

```bash
curl -sS -X POST http://localhost:8000/ask \
	-H "Content-Type: application/json" \
	-d '{
		"question": "What supply-chain risks did Apple describe in its most recent 10-K?",
		"k": 5
	}'
```

Example format only (not real output):

```json
{
	"answer": "<answer returned by the API>",
	"citations": [
		{"chunk_id": "<chunk id>", "quote": "<quoted filing text>", "verified": true}
	],
	"attempts": 1,
	"insufficient_context": false,
	"latency_ms": 1832.42
}
```

### 5. Run the smoke evaluation

```bash
python evals/smoke_questions.py --api-url http://localhost:8000
```

The script asks 10 AAPL/MSFT 10-K questions, prints the answer and citation verification details for each, and ends with a summary of insufficient-context rate, citation verification rate, and mean latency.

## Notes

- `ANTHROPIC_API_KEY` is required for the default LLM provider. `EDGAR_IDENTITY` must contain your name and email for SEC filing retrieval.
- If Docker is unavailable in your environment, run `docker compose config` to validate the compose file syntax, but do not claim live container startup unless it was actually tested.