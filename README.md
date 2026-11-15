# Advanced Wikipedia RAG
### Production-Oriented Retrieval-Augmented Generation with Hybrid Search, Cross-Encoder Reranking, Local LLM Inference & FastAPI

**Python | LlamaIndex | FAISS | BM25 | Reciprocal Rank Fusion | Cross-Encoder | Ollama | FastAPI | Streamlit | Pytest | GitHub Actions**

An end-to-end Retrieval-Augmented Generation (RAG) system designed to explore reliable, explainable, and modular LLM-powered question answering.

Unlike a basic semantic-search chatbot, this implementation combines dense and sparse retrieval, rank fusion, transformer-based reranking, source-grounded generation, REST API integration, automated testing, and retrieval evaluation.

The application runs its language model locally through Ollama and retrieves supporting knowledge from an indexed Wikipedia corpus.

---

## 1. Project Overview

Large Language Models can generate fluent responses but may produce unsupported claims or struggle to retrieve relevant information from a specific knowledge base.

This project addresses these limitations through an advanced RAG pipeline that retrieves external knowledge before generating a response.

### Engineering Objectives

- Develop a modular retrieval architecture combining semantic and lexical search.
- Improve candidate selection through Reciprocal Rank Fusion and cross-encoder reranking.
- Generate answers grounded in retrieved context rather than relying solely on parametric knowledge.
- Provide interpretable evidence through retrieved passages and source metadata.
- Expose inference functionality through a REST API.
- Support fully local LLM inference using Ollama.
- Establish reproducible testing and retrieval evaluation workflows.

**Project category:** Generative AI / LLM Engineering / Information Retrieval / Applied NLP

---

## 2. System Architecture

```text
                 USER QUESTION
                       |
                       v
                Query Processing
                       |
            +----------+----------+
            |                     |
            v                     v
       Dense Retrieval       Sparse Retrieval
       FAISS Index           BM25 Retriever
       Ollama Embeddings     Lexical Matching
            |                     |
            +----------+----------+
                       |
                       v
             Reciprocal Rank Fusion
                       |
                       v
               Candidate Passages
                       |
                       v
             Cross-Encoder Reranking
                       |
                       v
                Top-K Documents
                       |
                       v
                Context Assembly
                       |
                       v
             Local LLM (Llama 3.2)
                       |
                       v
              Grounded Generation
                       |
                       v
             Answer + Source Evidence
                       |
             +---------+---------+
             |                   |
             v                   v
        FastAPI API        Streamlit Interface
```

### Core Pipeline

**Query → Dense Search + BM25 → Rank Fusion → Reranking → Context Retrieval → LLM Generation → Grounded Answer**

The architecture separates retrieval from generation, allowing individual components to be evaluated and improved independently.

---

## 3. Advanced Retrieval Engineering

### 3.1 Dense Semantic Retrieval — FAISS

The dense retrieval component uses vector embeddings to represent text passages in a high-dimensional semantic space.

**Embedding model:** `nomic-embed-text`

**Vector indexing:** Facebook AI Similarity Search (FAISS)

The Wikipedia knowledge base is split into smaller text chunks. Each chunk is converted into a numerical embedding and stored in a vector index.

At query time:

1. The user's question is embedded using the same embedding model.
2. The resulting vector is compared against indexed document embeddings.
3. Semantically relevant passages are retrieved.

Dense retrieval can identify conceptually related information even when the query and document use different vocabulary.

**Engineering considerations:**

- Embedding consistency between indexing and inference
- Chunk size and overlap
- Vector similarity and top-K retrieval
- Persistent index loading
- Search latency and memory utilization

### 3.2 Sparse Retrieval — BM25

Semantic similarity alone may fail to prioritize exact keywords, technical terms, or uncommon phrases.

To address this, BM25 keyword retrieval is integrated alongside dense retrieval.

BM25 estimates document relevance using factors including:

- Query term frequency
- Inverse document frequency
- Document length normalization
- Term saturation

This makes sparse retrieval especially valuable for terminology-sensitive information retrieval.

### 3.3 Hybrid Search — Reciprocal Rank Fusion

The retrieval system combines results from FAISS and BM25 using Reciprocal Rank Fusion (RRF).

For a document \(d\):

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

Where:

- \(R\): Set of retrievers
- \(k\): Rank-fusion smoothing constant
- \(\operatorname{rank}_r(d)\): Rank of document \(d\) in retriever \(r\)

RRF provides a ranking-based method of merging retrieval results without requiring the raw dense and sparse scores to share the same scale.

**Why hybrid retrieval?**

Dense retrieval captures semantic similarity, whereas BM25 captures lexical relevance.

Combining both provides complementary retrieval signals.

### 3.4 Transformer-Based Cross-Encoder Reranking

After hybrid retrieval, candidate passages are passed through a transformer-based cross-encoder reranking stage.

**Reranking model:** `cross-encoder/ms-marco-MiniLM-L6-v2`

Unlike bi-encoder retrieval, which represents queries and documents independently, a cross-encoder processes the query and passage jointly.

This enables deeper query-document relevance estimation.

**Retrieval strategy:**

1. Retrieve candidate passages using dense and sparse search.
2. Merge the results through RRF.
3. Score candidates using the cross-encoder.
4. Select the highest-ranking passages for answer generation.

**Design trade-off:** Cross-encoder inference is computationally more expensive than vector similarity, so it is applied to a candidate shortlist rather than the complete corpus.

---

## 4. Retrieval-Augmented Answer Generation

The generation component uses a locally hosted language model.

**LLM:** `llama3.2:3b`

**Inference runtime:** Ollama

**Framework:** LlamaIndex

### Grounded Generation

Retrieved passages are used as reference context for answer generation.

The application is designed to constrain responses to the retrieved evidence and return an insufficient-information response when the available context does not support an answer.

This reduces reliance on unsupported model-generated information, although it does not guarantee factual correctness.

### Source Transparency

Responses are accompanied by retrieved source metadata, including Wikipedia article titles, links, and supporting passages.

This allows users to independently inspect the evidence.

### Out-of-Scope Questions

The system is intentionally limited to its indexed knowledge base.

For example:

**Question:** What is supervised machine learning?

**Expected behavior:** Retrieve relevant Wikipedia passages and generate a source-grounded explanation.

**Question:** What is the capital of Australia?

**Expected behavior:** Return insufficient information when the retrieved corpus does not contain the answer.

The latter behavior is intentional: the system prioritizes evidence-supported answers over unrestricted model recall.

---

## 5. Knowledge Base and Indexing

### Data Source

Wikipedia articles retrieved through the MediaWiki API.

**Initial knowledge domains:**

- Artificial intelligence
- Machine learning

### Indexing Workflow

```text
Wikipedia API
      |
      v
Article Extraction
      |
      v
Document Metadata
      |
      v
Text Chunking
      |
      v
Ollama Embeddings
      |
      v
FAISS Vector Index
      |
      v
Persistent Storage
```

The indexing workflow separates data preparation from online query inference.

Persistent storage enables the application to reuse an existing index rather than generating all embeddings on each startup.

The project includes `build_faiss.py` for creating the vector index.

---

## 6. REST API Engineering

The RAG pipeline is exposed through FastAPI to support programmatic access and integration with other applications.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | API health check |
| GET | `/ready` | Application readiness check |
| POST | `/query` | RAG question answering |
| GET | `/docs` | Interactive Swagger documentation |

### Example Request

**POST `/query`**

```json
{
  "question": "What is supervised machine learning?"
}
```

### Response Structure

The query endpoint returns a JSON response containing the question, generated answer, and retrieved source records.

Each source provides supporting information such as article metadata and an excerpt.

### API Design Considerations

- Request validation
- Structured JSON responses
- Consistent HTTP error handling
- Readiness and health endpoints
- Separation of API and retrieval logic
- Testable application boundaries

---

## 7. Interactive Application

The project provides a Streamlit-based user interface for interacting with the retrieval system.

### Interface Capabilities

- Natural-language question input
- AI-generated answers
- Retrieved Wikipedia references
- Expandable source passages
- Visibility into the local LLM and retrieval components
- Explicit handling of unsupported questions

The UI is separated from the underlying retrieval architecture so that the application can also be accessed through its FastAPI backend.

---

## 8. Retrieval Evaluation Framework

The repository includes an evaluation script and a structured evaluation dataset for comparing retrieval strategies.

**Evaluation file:** `evaluate.py`

**Dataset:** `evaluation_dataset.json`

### Retrieval Strategies

| Configuration | Retrieval mechanism |
|---|---|
| Baseline | Dense vector retrieval |
| Hybrid | FAISS + BM25 + RRF |
| Reranked | Hybrid retrieval + cross-encoder |

### Evaluation Metrics

**Recall@K**

Measures whether relevant documents are included among the top-K retrieved results.

\[
Recall@K = \frac{\text{Relevant documents retrieved in top K}}{\text{Total relevant documents}}
\]

**Mean Reciprocal Rank (MRR)**

Measures how highly the first relevant document is ranked.

\[
MRR = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

**Latency**

Measures retrieval or end-to-end processing time.

### Experimental Results

The evaluation pipeline is implemented, but final benchmark values should be reported only after checking the generated evaluation outputs.

| Metric | Dense | Hybrid | Reranked |
|---|---|---|---|
| Recall@K | Pending | Pending | Pending |
| MRR | Pending | Pending | Pending |
| Retrieval latency | Pending | Pending | Pending |

**Evaluation principle:** Improvements should be demonstrated through reproducible measurements rather than assumed from architectural complexity.

The initial Wikipedia corpus and evaluation dataset are limited in scope; larger and more diverse benchmarks would be required to establish generalizable retrieval improvements.

---

## 9. Automated Testing and Reliability

The application includes a Pytest-based API test suite.

**Test file:** `test_api.py`

### Verified Local Tests

Eight automated API tests passed during local development.

The test suite covers:

- Health endpoint behavior
- Empty question validation
- Whitespace-only questions
- Missing request fields
- Excessively long questions
- Successful query responses
- Query inference failures
- Additional API request handling

### Testing Strategy

The test suite uses mocked inference behavior where appropriate, allowing API-level functionality to be tested without requiring full model execution for every test.

### Run Tests

```bash
python -m pytest test_api.py -v
```

**Last verified local result: 8 tests passed.**

This verifies tested API behavior, not comprehensive end-to-end LLM accuracy.

---

## 10. Continuous Integration — GitHub Actions

The repository includes a GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The workflow is configured to:

1. Trigger on pushes and pull requests.
2. Checkout repository code.
3. Configure the Python environment.
4. Install project dependencies.
5. Execute automated API tests.
6. Report pass/fail status.

### CI Workflow

```text
GitHub Push / Pull Request
             |
             v
      GitHub Actions
             |
             v
       Python Setup
             |
             v
    Dependency Installation
             |
             v
       Pytest Execution
             |
             v
       Test Results
```

**CI status:** Workflow prepared; remote execution must be verified in the GitHub Actions tab.

Continuous integration helps identify regressions and dependency problems before code changes are merged.

---

## 11. Containerization

Docker configuration files are provided:

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`

The container configuration is designed to run the FastAPI application and connect it to Ollama on the host machine.

A Docker image has been created locally. However, full container startup and endpoint validation have not yet been completed due to Docker Desktop engine errors.

**Containerization status: Configured; runtime validation pending.**

This distinction is intentional to keep the repository's implementation claims accurate.

---

## 12. Project Structure

```text
wikipedia-rag/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- main.py
|-- api.py
|-- build_faiss.py
|-- evaluate.py
|-- test_api.py
|
|-- evaluation_dataset.json
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- Dockerfile
|-- .dockerignore
|-- docker-compose.yml
|
|-- wiki_faiss_index/       # Generated locally
`-- wiki_rag_ollama_v1/    # Original persisted index
```

Generated indexes and local environments should be excluded from version control.

---

## 13. Local Development Setup

### Prerequisites

- Python 3.13-compatible environment
- Ollama
- Git
- Required Python dependencies
- Internet access for initial Wikipedia article retrieval

### Clone Repository

```bash
git clone https://github.com/jayytisrivastav/wikipedia-rag.git
cd wikipedia-rag
```

### Create Virtual Environment

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Prepare Ollama Models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Ensure the Ollama service is running.

### Build the FAISS Index

```bash
python build_faiss.py
```

This step prepares the persistent vector index.

### Run Streamlit

```bash
python -m streamlit run main.py
```

### Run FastAPI

In a separate terminal:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Access the interactive API documentation:

`http://127.0.0.1:8000/docs`

### Run Automated Tests

```bash
python -m pytest test_api.py -v
```

### Run Retrieval Evaluation

```bash
python evaluate.py
```

Consult the evaluation script output for the generated benchmark results.

---

## 14. Engineering Decisions and Trade-offs

| Engineering decision | Rationale | Trade-off |
|---|---|---|
| Local Ollama inference | Local execution and model control | CPU inference may be slower |
| FAISS retrieval | Efficient vector similarity search | Requires embedding and indexing |
| BM25 integration | Exact lexical relevance | Limited semantic understanding |
| Reciprocal Rank Fusion | Combines complementary rankings | Adds retrieval complexity |
| Cross-encoder reranking | Richer query-document relevance modeling | Additional inference latency |
| Persistent indexes | Avoids rebuilding embeddings on every query | Requires index management |
| FastAPI backend | Programmatic access and typed interfaces | Adds a service layer |
| Automated API tests | Faster regression detection | Mocked tests do not measure answer quality |

### Key Technical Challenges

**Retrieval quality:** Semantic similarity and exact keyword relevance provide different signals. Hybrid search combines both.

**Answer grounding:** Language models may generate unsupported content. Retrieved context and source transparency provide mechanisms for evidence-based answering.

**Latency:** Adding a reranking stage can improve candidate ordering but increases computational cost.

**Reproducibility:** Local model availability, package versions, persisted indexes and OS-specific dependencies must be controlled for consistent execution.

---

## 15. Limitations and Future Roadmap

The current system demonstrates a modular RAG architecture, but additional work is needed before production deployment.

### Planned Improvements

- Expand the Wikipedia corpus beyond the initial AI/ML articles.
- Implement incremental indexing and document updates.
- Add larger human-labeled retrieval benchmarks.
- Evaluate answer faithfulness and source attribution.
- Introduce inference latency and resource monitoring.
- Add request tracing and retrieval observability.
- Improve failure recovery and model availability checks.
- Complete Docker runtime verification.
- Add security controls, rate limiting and deployment monitoring.
- Benchmark system throughput and concurrent requests.

These are future improvements rather than completed capabilities.

---

## 16. Skills Demonstrated

**LLM & Generative AI Engineering**
- Retrieval-Augmented Generation
- Context-grounded LLM prompting
- Local inference with Ollama
- LlamaIndex integration

**Information Retrieval**
- Dense vector search
- BM25 sparse retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Cross-encoder reranking

**Machine Learning Evaluation**
- Retrieval benchmarking
- Recall@K
- Mean Reciprocal Rank
- Latency analysis

**Software & API Engineering**
- Python
- FastAPI
- REST API development
- Request validation
- Error handling
- Pytest
- GitHub Actions
- Docker configuration

---

## 17. Conclusion

This project demonstrates the design and implementation of a modular RAG system that integrates modern information retrieval techniques with local language-model inference.

Its primary engineering contribution is the integration of **dense retrieval, sparse retrieval, reciprocal rank fusion, cross-encoder reranking, source-grounded generation, API development, and automated testing** into a single extensible application.

The repository provides a foundation for further work in production-oriented LLM systems, retrieval optimization, model evaluation, and AI application engineering.

---

**Author:** Jayyti Srivastav

**GitHub:** [jayytisrivastav](https://github.com/jayytisrivastav)

**Focus:** AI Engineering · Machine Learning · Generative AI · Computer Vision · Computational Imaging
