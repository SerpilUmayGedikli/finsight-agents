from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"

QUESTIONS = [
    "What are the biggest supply-chain risks described by Apple in its 10-K?",
    "What are the biggest supply-chain risks described by Microsoft in its 10-K?",
    "What risk factors does Apple cite related to competition and market demand?",
    "What risk factors does Microsoft cite related to competition and market saturation?",
    "How does Apple describe revenue concentration by segment or geography in its 10-K?",
    "How does Microsoft describe revenue concentration by segment or geography in its 10-K?",
    "What cybersecurity risks or incidents are discussed in Apple or Microsoft 10-K filings?",
    "What AI-related risks are disclosed by Apple or Microsoft in their 10-K filings?",
    "What legal proceedings or regulatory matters are material to Apple or Microsoft?",
    "What manufacturing, product, or operational disruptions are highlighted in the Apple and Microsoft 10-Ks?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the /ask endpoint against Apple and Microsoft 10-K questions."
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", DEFAULT_API_URL),
        help="Base URL of the API, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds for each /ask call.",
    )
    return parser.parse_args()


def print_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        print("  citations: []")
        return

    print("  citations:")
    for citation in citations:
        chunk_id = citation.get("chunk_id", "unknown")
        verified = bool(citation.get("verified", False))
        quote = citation.get("quote", "")
        print(
            f"    - chunk_id={chunk_id!r}, verified={verified}, quote={quote[:200]!r}"
        )


def summarize(results: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0

    insufficient_context = sum(
        1 for result in results if bool(result.get("insufficient_context"))
    )
    total_citations = sum(len(result.get("citations", [])) for result in results)
    verified_citations = sum(
        1
        for result in results
        for citation in result.get("citations", [])
        if bool(citation.get("verified", False))
    )
    mean_latency = statistics.fmean(
        float(result.get("latency_ms", 0.0)) for result in results
    )

    insufficient_pct = (insufficient_context / len(results)) * 100.0
    verified_pct = (
        (verified_citations / total_citations * 100.0) if total_citations else 0.0
    )
    return insufficient_pct, verified_pct, mean_latency


def main() -> int:
    args = parse_args()
    api_url = args.api_url.rstrip("/")
    client = httpx.Client(timeout=args.timeout)
    results: list[dict[str, Any]] = []

    for index, question in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        try:
            response = client.post(
                f"{api_url}/ask",
                json={"question": question, "k": 5},
            )
            response.raise_for_status()
        except httpx.RequestError as exc:
            print(f"API unreachable at {api_url}: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPStatusError as exc:
            print(
                f"Request failed for question {index}: {exc.response.status_code} {exc.response.text}",
                file=sys.stderr,
            )
            return 1

        payload = response.json()
        latency_ms = float(
            payload.get("latency_ms", (time.perf_counter() - started) * 1000.0)
        )
        citations = payload.get("citations", [])
        attempts = int(payload.get("attempts", 0))
        answer = str(payload.get("answer", ""))
        insufficient_context = bool(payload.get("insufficient_context", False))

        print(f"\n[{index}] Question: {question}")
        print(f"  answer: {answer}")
        print_citations(citations)
        print(f"  latency_ms: {latency_ms:.2f}")
        print(f"  attempts: {attempts}")
        print(f"  insufficient_context: {insufficient_context}")

        results.append(
            {
                "answer": answer,
                "citations": citations,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "insufficient_context": insufficient_context,
            }
        )

    insufficient_pct, verified_pct, mean_latency = summarize(results)
    print("\nSummary:")
    print(f"  insufficient_context: {insufficient_pct:.1f}%")
    print(f"  citations_verified: {verified_pct:.1f}%")
    print(f"  mean_latency_ms: {mean_latency:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
