# Wikipedia RAG with Local LLMs

A Python application for answering questions using selected Wikipedia
articles, with a Streamlit interface and local inference through Ollama.

## Features
- Wikipedia article retrieval over HTTPS.
- Text chunking with overlapping passages.
- Semantic retrieval using nomic-embed-text embeddings.
- Answer generation using Llama 3.2 3B.
- Persistent vector index and cached query engine.
- Retrieved passages and Wikipedia links for checking answers.

## Technologies
Python, LlamaIndex, Ollama, Streamlit, Requests.

## Run locally
Install Ollama, then download the models:

    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Install the Python dependencies:

    python -m pip install -r requirements.txt

Keep Ollama running and launch the application:

    python -m streamlit run main.py

The first question downloads and indexes the Wikipedia articles.
The saved index is reused on subsequent runs.

## Scope and limitations
Currently indexes the Artificial intelligence and Machine learning
Wikipedia articles. Responses may contain errors; retrieved passages
should be checked. No formal retrieval or answer-quality evaluation
has been completed.

## Acknowledgements
Inspired by NeuralNine's Wikipedia RAG tutorial and adapted for
local Ollama models, HTTPS article retrieval and source inspection.
Wikipedia content belongs to its respective contributors and is
subject to the licensing terms on the source articles.