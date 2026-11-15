
from pathlib import Path

import faiss

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from main import download_articles, OLLAMA_URL, EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "wiki_faiss_index"


def main():
    embedding_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )

    documents = download_articles()

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Read the embedding dimension from the actual model.
    sample_embedding = embedding_model.get_text_embedding(
        "Wikipedia FAISS index"
    )
    dimension = len(sample_embedding)

    # Flat L2 index: exact nearest-neighbor baseline.
    faiss_index = faiss.IndexFlatL2(dimension)

    vector_store = FaissVectorStore(
        faiss_index=faiss_index
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True,
    )

    storage_context.persist(
        persist_dir=str(FAISS_DIR)
    )

    print("FAISS index created successfully.")
    print(f"Indexed nodes: {len(nodes)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Saved at: {FAISS_DIR}")


if __name__ == "__main__":
    main()
