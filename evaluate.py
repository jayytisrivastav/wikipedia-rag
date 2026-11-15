
import json
import time
from pathlib import Path
from statistics import mean

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.retrievers.bm25 import BM25Retriever

from main import (
    load_vector_index,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "evaluation_dataset.json"

QUESTIONS = [
    "What is artificial intelligence?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",
    "What is the difference between AI and ML?",
]


def create_retrievers(index):
    dense = index.as_retriever(similarity_top_k=5)

    bm25 = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=5,
    )

    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=300.0,
    )

    hybrid = QueryFusionRetriever(
        retrievers=[dense, bm25],
        llm=llm,
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L6-v2",
        top_n=3,
        device="cpu",
    )

    return dense, hybrid, reranker


def retrieve(query, pipeline, dense, hybrid, reranker):
    start = time.perf_counter()

    if pipeline == "Dense":
        nodes = dense.retrieve(query)[:3]

    elif pipeline == "Hybrid":
        nodes = hybrid.retrieve(query)[:3]

    else:
        candidates = hybrid.retrieve(query)
        nodes = reranker.postprocess_nodes(
            candidates,
            query_str=query,
        )[:3]

    latency_ms = (time.perf_counter() - start) * 1000

    ids = [item.node.node_id for item in nodes]
    return nodes, ids, latency_ms


def prepare_dataset(dense, hybrid, reranker):
    dataset = []

    for question in QUESTIONS:
        candidates = {}

        for pipeline in ["Dense", "Hybrid", "Reranked"]:
            nodes, _, _ = retrieve(
                question, pipeline, dense, hybrid, reranker
            )

            for item in nodes:
                node = item.node
                candidates[node.node_id] = {
                    "node_id": node.node_id,
                    "title": node.metadata.get("title", ""),
                    "excerpt": node.get_content()[:500],
                }

        dataset.append({
            "question": question,
            "expected_node_ids": [],
            "candidates": list(candidates.values()),
        })

    DATASET_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nCreated: {DATASET_FILE.name}")
    print("Review the candidate excerpts for each question.")
    print("Copy relevant node_id values into expected_node_ids.")
    print("Then run: python evaluate.py")


def calculate_metrics(retrieved_ids, expected_ids):
    relevant = set(expected_ids)
    hits = [
        rank
        for rank, node_id in enumerate(retrieved_ids, start=1)
        if node_id in relevant
    ]

    hit_at_3 = 1.0 if hits else 0.0
    recall_at_3 = len(hits) / len(relevant)
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0

    return hit_at_3, recall_at_3, reciprocal_rank


def evaluate(dense, hybrid, reranker):
    dataset = json.loads(
        DATASET_FILE.read_text(encoding="utf-8")
    )

    if not dataset or any(
        not item.get("expected_node_ids")
        for item in dataset
    ):
        print(
            "Please label expected_node_ids for every "
            "question in evaluation_dataset.json."
        )
        return

    results = {
        name: []
        for name in ["Dense", "Hybrid", "Reranked"]
    }

    for item in dataset:
        question = item["question"]
        expected = item["expected_node_ids"]

        print(f"\nQuestion: {question}")

        for pipeline in results:
            _, retrieved_ids, latency = retrieve(
                question,
                pipeline,
                dense,
                hybrid,
                reranker,
            )

            hit, recall, mrr = calculate_metrics(
                retrieved_ids,
                expected,
            )

            results[pipeline].append({
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "latency_ms": latency,
            })

    print("\nRETRIEVAL BENCHMARK RESULTS")
    print("-" * 76)
    print(
        f"{'Pipeline':<14} {'Hit@3':>9} "
        f"{'Recall@3':>12} {'MRR':>9} "
        f"{'Latency(ms)':>15}"
    )

    summary = []

    for pipeline, rows in results.items():
        result = {
            "pipeline": pipeline,
            "hit_at_3": mean(r["hit"] for r in rows),
            "recall_at_3": mean(r["recall"] for r in rows),
            "mrr": mean(r["mrr"] for r in rows),
            "mean_latency_ms": mean(
                r["latency_ms"] for r in rows
            ),
        }
        summary.append(result)

        print(
            f"{pipeline:<14} "
            f"{result['hit_at_3']:>9.3f} "
            f"{result['recall_at_3']:>12.3f} "
            f"{result['mrr']:>9.3f} "
            f"{result['mean_latency_ms']:>15.1f}"
        )

    output_path = BASE_DIR / "evaluation_results.json"
    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to {output_path.name}")


def main():
    print("Loading Wikipedia vector index...")

    embedding_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    index = load_vector_index(embedding_model)

    print("Initializing retrieval pipelines...")
    dense, hybrid, reranker = create_retrievers(index)

    if not DATASET_FILE.exists():
        prepare_dataset(dense, hybrid, reranker)
    else:
        evaluate(dense, hybrid, reranker)


if __name__ == "__main__":
    main()
