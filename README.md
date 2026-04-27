<p align="center">
  <img src="nare_banner.png" alt="NARE Framework Banner" width="850"/>
</p>

<h1 align="center">NARE: Non-parametric Amortized Reasoning Evolution</h1>

<p align="center">
  <em>An Experimental Cognitive Architecture for Deterministic Routing and Memory Crystallization</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental_research-orange.svg?style=flat-square" alt="Status: Experimental">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Code style: black"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
</p>

> 🚧 **Project Status: Research Preview / Pre-Alpha**
> NARE is currently an independent, experimental research initiative. It is **not** yet production-ready. Core architectures, memory structures, and routing protocols are in active development and subject to frequent breaking changes.

---

## Executive Summary

**NARE (Neuro-Adaptive Reasoning Engine)** is an experimental cognitive framework designed to explore the transition of computationally expensive LLM reasoning (System 2) into zero-shot, deterministic execution (System 1). By continuously monitoring its own reasoning trajectories, NARE tests a "sleep phase" consolidation mechanism aimed at crystallizing recurring logical patterns into secure, executable Python abstractions.

In early conceptual testing, this amortized inference model shows potential for an **O(1) latency footprint** and **drastically reduced API reliance** for recurring task classes.

---

## System Architecture

### 1. The 4-Way Routing Protocol
NARE dynamically routes incoming queries in an attempt to minimize expected free energy and computational cost.

| Route | Execution Environment | Target Latency | Token Cost | Condition |
| :--- | :--- | :--- | :--- | :--- |
| **REFLEX** | Procedural (Python AST Sandbox) | ~1 ms | 0 | Mature skill, high confidence |
| **FAST** | Vector Retrieval (FAISS) | ~10 ms | 0 | Exact semantic match ($sim \ge 0.98$) |
| **HYBRID** | Context-Augmented LLM | ~2000 ms | ~15% | Partial match, $\delta$-reasoning required |
| **SLOW** | Tree-of-Thoughts (ToT) + Critic | > 60 s | 100% | Novel problem, low confidence |

### 2. Six-Tier Memory Hierarchy
The engine iterates on a multi-modal memory system designed to support lifelong learning:

* **Episodic:** Dense vector indexing (FAISS IndexFlatIP) for trajectory caching with Ebbinghaus decay.
* **Semantic:** Crystallized Python AST functions representing compiled algorithmic skills.
* **Factual (RAG):** Knowledge base retrieval with strict deduplication heuristics.
* **Graph:** Associative structures governed by Hebbian strengthening.
* **RL Retriever:** Contextual bandit leveraging $\epsilon$-greedy exploration for context injection.
* **Neural (Titans):** Online MLP implementation with surprise-driven gating.

### 3. Sleep Phase Crystallization
Memory consolidation occurs offline, translating high-dimensional reasoning paths into deterministic code.

* **NREM Phase:** Aggregation of episodic memories via FAISS clustering; initial code synthesis.
* **REM Phase:** Adversarial stress-testing, edge-case generation, and iterative self-repair.
* **Meta-Abduction:** Early exploration of structural isomorphisms to formulate cross-domain abstract principles.

---

## Quickstart Guide

### Prerequisites
* Python 3.10 or higher.
* Google Gemini API Key ([Acquire here](https://aistudio.google.com/apikey)).

### Installation & Configuration

```bash
git clone [https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine.git](https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine.git)
cd Neuro-Adaptive-Reasoning-Engine

pip install -r requirements.txt

# Environment configuration
echo "GEMINI_API_KEY=your_api_key" > .env
