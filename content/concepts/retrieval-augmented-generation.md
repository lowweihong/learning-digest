---
title: Retrieval-Augmented Generation
aliases: [RAG, external retrieval grounding, vector database augmentation, dense passage retrieval]
tags: [concept, llm, rag, information-retrieval, agentic-ai]
---

## Definition
A system architecture that enhances large language model generation by retrieving relevant factual documents from an external knowledge base (such as a vector database or search index) and injecting them into the model's prompt context, grounding responses in verifiable, up-to-date source material without modifying model parameters.

## Why it's true
Pre-trained large language models store vast amounts of world knowledge implicitly inside their billions of connection weights. However, relying purely on parametric memory suffers from three fundamental flaws:
1. **Knowledge Staleness:** Information changes constantly (company documentation, legal statutes, medical studies, live stock prices). Re-training or fine-tuning foundation models to absorb new facts is computationally expensive and slow.
2. **Hallucination:** When an LLM cannot recall a precise detail from its pre-training corpus, its autoregressive objective generates statistically plausible but factually fictitious hallucinations.
3. **Black-Box Opacity:** Pure parametric generation cannot provide clickable, auditable citations back to verified primary source documents.

**Retrieval-Augmented Generation (RAG)** decouples **reasoning** (performed by the LLM) from **knowledge storage** (managed by external databases):
1. **Document Ingestion & Indexing:** Unstructured documents (PDFs, wikis, codebases) are split into semantic chunks and passed through an embedding model to generate high-dimensional vectors, which are indexed inside a vector database (e.g., Faiss, Milvus, Qdrant).
2. **Dense Retrieval:** When a user submits a query $q$, the query is embedded into the same latent space, and approximate nearest neighbor (ANN) search identifies the top-$k$ most semantically relevant document chunks.
3. **Reranking (Optional):** A cross-encoder reranking model scores the candidate chunks against the query to filter out irrelevant passages, optimizing context window token usage.
4. **Context Injection:** The retrieved passages are packaged alongside the user's question into an augmented prompt:
   ```
   Context information is below:
   [Retrieved Document 1]
   [Retrieved Document 2]
   Given the context above, answer the question: [User Query]
   ```
5. **Grounded Generation:** The LLM synthesizes an accurate, factual response conditioned strictly on the retrieved evidence, including verbatim quotes and source citations.

## Minimal example
An enterprise software question: *"What is our company's refund policy for cloud subscriptions after 14 days?"*
- **Base LLM (Parametric only):** Generates a generic, fabricated answer based on public internet policies, which is legally inaccurate for the specific enterprise.
- **RAG Pipeline:**
  1. The vector database matches the query to Clause 4.2 of the enterprise legal terms of service.
  2. The raw text of Clause 4.2 is injected into the context window.
  3. The LLM responds: *"According to Section 4.2 of the Enterprise Service Agreement, subscriptions past 14 days are non-refundable, but eligible for prorated service credits upon manager approval."*
The answer is grounded, verifiable, and required zero model fine-tuning.

## Prerequisites
- None

## Common confusions
- **RAG vs. Fine-Tuning:** Fine-tuning teaches a model **how to behave, communicate, and format** (teaching a style, tone, or reasoning methodology). RAG teaches a model **what facts to know** (supplying specific up-to-date information). For factual knowledge retrieval, RAG is vastly superior and more cost-effective than fine-tuning.
- **RAG eliminates all hallucinations:** If the retriever fetches irrelevant or conflicting document chunks, the language model can still misinterpret the context. High retrieval recall and precision (via rerankers and hybrid keyword/dense search) are critical to maintaining accuracy.

## Appears in
- [[lessons/2026-11-01-llm-post-training-and-alignment]]: introduced as the foundational architecture connecting language models to external non-parametric memory.
