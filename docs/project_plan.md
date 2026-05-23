# Project Plan

## Objective

Build a lightweight offline RAG assistant that helps CNC operators find accurate answers from machine manuals.

## MVP Scope

- Target one machine family: Haas CNC mill with Next Generation Control.
- Ingest one official operator manual PDF.
- Extract text with page metadata.
- Chunk text into retrieval-friendly passages.
- Build a local FAISS vector index.
- Support CLI search with citations.
- Support optional answer generation through a local Ollama model.

## Non-Goals For MVP

- No direct machine integration.
- No live sensor/controller connection.
- No automatic execution of CNC commands.
- No proprietary manuals.
- No video transcription until the text RAG pipeline works.

## Evaluation Questions

Use questions like:

- How do I select the active program?
- What safety precautions apply before machine operation?
- How do I use MDI mode?
- What does the manual say about coolant or lubrication checks?
- Where can I find G-code or M-code reference information?

For each question, evaluate:

- Did retrieval return the correct manual section?
- Does the answer cite the source page?
- Does the answer avoid unsupported claims?
- How long did retrieval and generation take on local hardware?

## Portfolio Story

This project demonstrates practical industrial AI:

- Offline-first deployment for shop-floor environments.
- Local embeddings and vector search.
- Source-grounded answers with citations.
- Safety-aware response behavior.
- Measurable retrieval quality instead of generic chatbot claims.

