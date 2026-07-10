# GraphRAG — From Raw Text to a Knowledge-Graph-Backed Chatbot

![GitHub](https://img.shields.io/github/license/DataForScience/GraphRAG)
[![Twitter @data4sci](https://img.shields.io/twitter/follow/data4sci)](https://twitter.com/intent/follow?screen_name=data4sci)
![GitHub top language](https://img.shields.io/github/languages/top/DataForScience/GraphRAG)
![GitHub repo size](https://img.shields.io/github/repo-size/DataForScience/GraphRAG)
![GitHub last commit](https://img.shields.io/github/last-commit/DataForScience/GraphRAG)

[![Graphs For Science](https://img.shields.io/badge/Graphs_For_Science-Subscribe-blue)](https://graphs4sci.substack.com/)
	[![Data Science Briefing](https://img.shields.io/badge/Sunday_Briefing-Subscribe-blue)](https://data4science.ck.page/a63d4cc8d9)

### Code and slides to accompany the online series of webinars: https://data4sci.com/graphrag by Data For Science.

A four-hour, hands-on workshop that builds a **Graph RAG system from scratch** — no black boxes. Starting from raw Wikipedia text (wikitext-103), we assemble the full pipeline with classical, locally-runnable NLP models:

```
                     OFFLINE (index time)
raw text ─► spaCy ─► fastcoref ─► REBEL ─► triples ─► NetworkX graph
                                                          │
                     ONLINE (query time)                  ▼
question ─► entity linking ─► graph traversal ─► facts ─► LLM ─► grounded answer
```

Along the way we cover why vector RAG breaks on multi-hop, "global", and relationship-first questions; how knowledge graphs fix that by making facts first-class, traversable objects; how to evaluate retrieval and answer quality separately; and what changes in production (persistence, entity resolution, incremental updates, extraction drift, monitoring). The closing angle: the same graph doubles as **durable, auditable memory for LLM agents**.

Everything runs **in-process** — no database server, no Docker. The LLM (Anthropic API) is optional: without an API key, a template renders the retrieved facts directly, and the retrieval machinery (the actual lesson) is identical.

## Repository contents

| Path | What it is |
|---|---|
| `1. Foundations.ipynb` | Part 1 — RAG fundamentals, a TF-IDF baseline, Graph RAG architecture, environment setup, corpus loading + EDA. Writes `corpus.txt`. |
| `2. Text To Triples.ipynb` | Part 2 — Stage 1 spaCy (sentences + NER), Stage 2 fastcoref (coreference resolution), Stage 3 REBEL ((head, relation, tail) triples) + cleaning. Writes `entities.jsonl`, `resolved.txt`, `triples.jsonl`. |
| `3. Graph.ipynb` | Part 3 — Stage 4 NetworkX: build the graph, query it in plain Python (`facts_about`, `path_between`), PageRank, connected components, communities, ego-graph visualization. |
| `4. Chatbot.ipynb` | Part 4 — the graph-backed chatbot (entity linking → traversal → grounded generation), evaluation vs. the vector baseline, production considerations. |
| `slides/GraphRAG.pdf` | The full workshop slide deck (10 slides, with background deep-dives on RAG, knowledge graphs, NER, coreference, REBEL, graph algorithms, and evaluation). |
| `checkpoints/` | On-disk artifacts each notebook reads/writes (`corpus.txt` → `resolved.txt` → `triples.jsonl` → graph pickles). Created on first run. Downloadable as a zip file from [here](https://drive.google.com/file/d/1BtpEuDDOFPhjxXu5pCj_J5WTqph3ZDIw/view?usp=sharing) (~6BG)|

**The checkpoint system:** each notebook ends by saving its outputs to `checkpoints/` and begins by loading what it needs from there. If a checkpoint is missing — you skipped a part, or a model failed to download — the notebook falls back to built-in data and keeps going, so every part runs end to end, alone or in sequence. Long stages (NER, coref, REBEL) checkpoint incrementally: interrupt the kernel any time and re-run to resume.

## Contents

- Intro & roadmap

Why knowledge graphs (recommendations, fraud, drug discovery) and the agent-memory angle. Demo the finished chatbot, then walk the four-hour map so people know when the hands-on parts land.

- RAG fundamentals

The retrieval-augmented generation loop: chunk → embed → retrieve → generate. Why we ground LLMs at all (hallucination, stale knowledge, private data). Vector search and embeddings at a conceptual level, then the failure modes that motivate graphs: multi-hop questions, “global” questions spanning many documents, and queries that hinge on relationships rather than text similarity. A tiny vector-RAG snippet here makes a useful baseline to beat later.

- Graph RAG architecture

How graph RAG reframes retrieval: entities and typed relations as the retrieval substrate instead of opaque chunks. The architecture they’ll build — text → triples → graph → query → grounded generation — with a diagram. Contrast with vector RAG (and hybrid approaches), and note where the field sits (e.g., Microsoft’s GraphRAG, the broader “structured memory for agents” trend). Tie back to durable agent memory vs. ephemeral vector context.

- Knowledge graph construction overview

The construction problem end to end: from raw prose to (head, relation, tail) triples to nodes and edges. Why it’s four stages, what each one fixes, and where graphs typically break (spoiler: skipped coreference). Sets up the next three blocks.

- spaCy

Load a slice of wikitext-103-v1, sentence segmentation, and NER. Inspect entities on a sample paragraph so people see raw material before transformation.

- fastcoref

Resolve pronouns and references. The make-or-break step: show a before/after where unresolved coref produces orphaned or wrong nodes, and discuss why most quick pipelines skip it.

- REBEL

Extract (head, relation, tail) triples from the coref-resolved text. Eyeball the output, discuss noise, confidence thresholds, and a simple filtering pass.

- NetworkX, build & explore

Build a DiGraph (entities as nodes, relation stored as an edge attribute). Query it in plain Python — neighbors, paths, subgraph filtering — then run built-in algorithms (PageRank, connected components, community detection via greedy_modularity_communities or python-louvain) and visualize a subgraph with matplotlib or pyvis.

- Graph-backed chatbot

Wire it all together: natural-language question → the LLM identifies the relevant entity/relation → your Python code traverses NetworkX → the retrieved subgraph is fed back as grounding → answer. Run the full pipeline on a fresh paragraph so attendees see text become graph become answer in one pass. Contrast a grounded response with a hallucinated one.

- Evaluation

How to know it works. Split into retrieval quality (did we pull the right subgraph for the question?) and answer quality (faithfulness/groundedness, answer relevance). Build a small multi-hop question set the graph should answer but naive vector RAG struggles with, and compare the two. Mention tooling like RAGAS and LLM-as-judge, plus the honest limitations of each.

- Production considerations

What changes beyond a workshop slice: persisting the graph (GraphML/pickle vs. graduating to a server-backed store when it outgrows RAM), incremental updates as new documents arrive, entity resolution and deduplication at scale, extraction-quality drift from REBEL, latency and cost of the LLM-in-the-loop, and monitoring what the graph and chatbot actually return.

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies and launch Jupyter:

```bash
uv sync
uv run jupyter notebook
```

Notes:

- First run downloads models: spaCy `en_core_web_sm`, `biu-nlp/f-coref`, and REBEL (`Babelscape/rebel-large`, ~1.6 GB) — this is the main time sink, so run Part 1's setup cell early.
- GPU is optional. Pipelines auto-detect `cuda` / Apple `mps` / CPU; everything works on CPU, just slower. Use `load_wikitext(target_chars=4000)` in Part 1 for a quick run — REBEL costs ~1 s/sentence on CPU.
- For the chatbot's LLM generation step, set `ANTHROPIC_API_KEY` in your environment (optional).

## Author

<table border="0">
 <tr>
	<td>
	  <img src="data/bgoncalves.png" alt="Bruno Gonçalves" width="150" height="150" style="border-radius: 50%; object-fit: cover;">
	</td>
	<td>
	  <h2>Bruno Gonçalves</h2>
	  <h3>Data For Science, Inc.</h3>
	  <p>
			Web: <a href="http://www.data4sci.com/">www.data4sci.com</a><br>
			Twitter/X: <a href="https://twitter.com/bgoncalves">@bgoncalves</a><br>
			LinkedIn: <a href="https://www.linkedin.com/in/bmtgoncalves/">@bmtgoncalves</a><br>
			Email: <a href="info@data4sci.com">info@data4sci.com</a><br>
			Schedule a Call: <a href="https://data4sci.com/call">https://data4sci.com/call</a>
	  </p>
	</td>
 </tr>
</table>
