# CRAG Reproduction & Analysis on PopQA

Open-source reproduction and analysis of **Corrective Retrieval-Augmented Generation (CRAG)** on the PopQA benchmark, replacing the original fine-tuned retrieval evaluator with a **prompt-driven quantized model** running on a single free-tier GPU.

**Headline finding:** CRAG's benefit is *inversely related to the strength of the retriever it corrects*. Correction helps (and does no harm) when retrieval is weak, and turns negative once the local context is already reliable.

---

## Overview

Standard RAG assumes retrieved passages are relevant; when they aren't, the generator amplifies the error rather than correcting it. CRAG (Yan et al., 2024) adds a lightweight retrieval evaluator that labels retrieved documents as **Correct**, **Ambiguous**, or **Incorrect** and applies a corrective action — including a web-search fallback — before generation.

This project reproduces the full CRAG pipeline using only open-source components, then analyses *when* and *why* the corrective mechanism helps through ablations, retriever and chunk-size sweeps, and a manual error taxonomy.

## Key Results

**Vanilla RAG vs. CRAG** (dense retriever, calibrated grader; hard corpus adds embedding re-ranking)

| Corpus | Vanilla | CRAG | Δ | Fixed | Broke |
|--------|---------|------|------|-------|-------|
| Easy   | 61%     | 65%  | +4%  | 4     | 0     |
| Hard   | 65%     | 62%  | −3%  | 0     | 3     |

**Retriever sensitivity** — CRAG's advantage shrinks as the base retriever strengthens

| Retriever      | Vanilla | CRAG | Δ    |
|----------------|---------|------|------|
| BM25 (sparse)  | 20%     | 30%  | +10% |
| Dense (FAISS)  | 60%     | 64%  | +4%  |
| Hybrid         | 64%     | 66%  | +2%  |

**Chunk-size sensitivity** — delta turns negative once larger chunks make the baseline strong

| Chunk size (tokens) | Vanilla | CRAG | Δ   |
|---------------------|---------|------|-----|
| 128                 | 60%     | 62%  | +2% |
| 256                 | 60%     | 62%  | +2% |
| 512                 | 70%     | 64%  | −6% |

**Component ablation** — the grader is the highest-value component

| Configuration                       | Accuracy | Δ vs. full |
|-------------------------------------|----------|------------|
| Full CRAG                           | 63%      | —          |
| No corrective action for Ambiguous  | 60%      | −3%        |
| No corrective action for Incorrect  | 61%      | −2%        |
| No retrieval evaluator              | 59%      | −4%        |

**Error taxonomy (hard corpus)** — CRAG fails conservatively; refusals dominate, web-induced wrong answers are rare

| Category                       | % of errors |
|--------------------------------|-------------|
| Refusal ("I don't know")       | 68%         |
| Wrong answer from local context| 27%         |
| Wrong answer from web search   | 5%          |

> Prompt calibration of the grader alone moved the system by ~14 points. Strict substring exact-match scoring means these accuracies are conservative lower bounds.

## Repository Structure

| Notebook | Purpose |
|----------|---------|
| `CRAG_complete_pipeline.ipynb` | The full study end to end — setup, local model + embeddings, PopQA data and splits, all CRAG components (calibrated grader, rewriter, web search, generator, re-ranking), the LangGraph state machine, both corpora (easy / hard), and the Vanilla RAG vs. CRAG comparison. **Start here.** |
| `CRAG_chunksize_sweep.ipynb` | Chunk-size sensitivity analysis (128 / 256 / 512 tokens) for Vanilla RAG vs. CRAG. |
| `CRAG_ablation_study.ipynb` | Component ablation (Objective 3). Evaluator-quality ablation (strict vs. calibrated grader) and action ablation (full CRAG, w/o evaluator = vanilla, w/o Incorrect action, w/o Ambiguous action). |
| `CRAG_error_taxonomy.ipynb` | Auto-classifies every incorrectly answered question into a failure taxonomy (retrieval failure, extraction failure, hallucination, grader misroute, noisy gold), then supports hand-review and export to a labelled CSV. |
| `CRAG_gradio_demo_complete.ipynb` | Self-contained interactive Gradio demo. Pick a PopQA question and watch the pipeline run with the full decision trace: retrieved docs → grader verdict + reasoning → corrective branch → final answer + exact-match. |

## Pipeline

The components are wired as a conditional LangGraph state machine; each question's path is decided at inference time by the grader's verdict:

```
retrieve → grade → { Correct:            generate
                     Incorrect/Ambiguous: rewrite → web search → augment → re-rank → generate }
```

- **Retriever** — FAISS dense vector store over `BAAI/bge-small-en-v1.5` embeddings, BM25 sparse via `rank-bm25`, plus a hybrid configuration.
- **Grader** — prompt-driven; emits a small JSON object (one-sentence justification + categorical score) parsed directly, which is more robust on local quantized models than structured-output APIs. Calibrated to default to Correct/Ambiguous when uncertain, reserving Incorrect for clearly irrelevant retrievals.
- **Corrective branch** — context-aware query rewriter → DuckDuckGo web search (via `ddgs`) → augment local context → embedding-similarity re-rank → generate.

## Setup

Built and tested on a **free-tier Google Colab T4 GPU**.

**Backend model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (4-bit quantized, ~5.5 GB on disk, ~5.7 GB VRAM) — the pre-quantized checkpoint is loaded directly to avoid the much larger FP16 download.

```bash
pip install -r requirements.txt
```

Then open any notebook in Colab. The complete pipeline notebook loads the model, builds the corpus, and runs the experiments from scratch. Tip: set `HF_HOME` to a Google Drive path at session start to persist the model cache across runtime resets.

**Core libraries:** LangChain 0.3.x · LangGraph · FAISS-CPU · sentence-transformers · datasets · rank-bm25 · ddgs

## Dataset

[PopQA](https://huggingface.co/datasets/akariasai/PopQA) — 14,267 entity-centric QA pairs from Wikidata. Experiments use a fixed 100-question subset to stay tractable on the free tier while preserving statistical signal. Accuracy is substring exact-match: a prediction counts as correct if any gold answer string appears in the generated answer after normalization. (Note: `possible_answers` is a JSON-encoded string and is parsed with `json.loads` before scoring.)

Two corpora of increasing difficulty are built per experiment from MediaWiki article extracts:
- **Easy** — only the Wikipedia articles for the eval subjects. Retrieval nearly always succeeds; tests whether CRAG is *harmless* when retrieval is strong.
- **Hard** — eval articles plus ~300 distractors (≈4:1 distractor-to-signal). Produces genuine retrieval failures; this is where the corrective mechanism is actually exercised.

## Notes & Caveats

- Main comparisons rest on single 100-question runs, so deltas are directionally meaningful rather than precise effect sizes.
- The web fallback's coverage is the principal constraint on CRAG's gains — for entity-centric questions, search results frequently contain no usable evidence.

## References

1. Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884.
2. Mallen, A., et al. (2023). *When Not to Trust Language Models.* ACL 2023.
3. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33.
4. Asai, A., et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024.

---

*Mehwish Bibi — Istanbul Medipol University. Large Language Models, Final Project (Track 3: Analysis of an Existing Project).*
