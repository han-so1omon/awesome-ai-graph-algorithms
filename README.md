# Awesome AI Graph Algorithms (LLM × Graphs) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated awesome list of AI-powered graph algorithms and systems:

---

## Table of Contents
- [GraphRAG and Graph-Indexed RAG](#graphrag-and-graph-indexed-rag)
- [LLM ↔ Graph Learning](#llm--graph-learning)
- [LLM ↔ Knowledge Graph (KG)](#llm--knowledge-graph-kg)
- [Graph Storage and Compression (AI-informed dedup)](#graph-storage-and-compression-ai-informed-dedup)
- [Graph Reasoning Agents and Tool Use](#graph-reasoning-agents-and-tool-use)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [Graph Generation (LLM / Agents / Diffusion)](#graph-generation-llm--agents--diffusion)
- [Neural Algorithmic Graph Reasoning](#neural-algorithmic-graph-reasoning)
- [Production Graph ML Systems](#production-graph-ml-systems)
- [Libraries](#libraries)
- [Meta Lists and Surveys](#meta-lists-and-surveys)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Inspirations](#inspirations)

---

## GraphRAG and Graph-Indexed RAG

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **GraphRAG-R1** | ![Research](https://img.shields.io/badge/Type-Research-blue) | RAG | 2025-07 | 2025-07 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2507.23581 | — |
| **Retrieval-Augmented Generation with Graphs (GraphRAG)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | RAG | 2025-01 | 2025-01 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2501.00309 | — |
| **MiniRAG** | ![Research](https://img.shields.io/badge/Type-Research-blue) | RAG | 2025-01 | 2025-?? | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/HKUDS/MiniRAG?style=social) ![last](https://img.shields.io/github/last-commit/HKUDS/MiniRAG) | Paper: https://arxiv.org/abs/2501.06713 · Code: https://github.com/HKUDS/MiniRAG | — |
| **GFM-RAG** | ![Research](https://img.shields.io/badge/Type-Research-blue) | RAG / KG completion | 2025-02 | 2025-?? | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/RManLuo/gfm-rag?style=social) ![last](https://img.shields.io/github/last-commit/RManLuo/gfm-rag) | Paper: https://arxiv.org/abs/2502.01113 · Code: https://github.com/RManLuo/gfm-rag · Docs: https://rmanluo.github.io/gfm-rag/ | HF model: https://huggingface.co/rmanluo/GFM-RAG-8M |
| **LightRAG** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | RAG | 2024-10 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/HKUDS/LightRAG?style=social) ![last](https://img.shields.io/github/last-commit/HKUDS/LightRAG) | Paper: https://arxiv.org/abs/2410.05779 · Code: https://github.com/HKUDS/LightRAG | — |
| **Microsoft GraphRAG** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | RAG | 2024-04 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/microsoft/graphrag?style=social) ![last](https://img.shields.io/github/last-commit/microsoft/graphrag) | Paper: https://arxiv.org/abs/2404.16130 · Code: https://github.com/microsoft/graphrag · Docs: https://microsoft.github.io/graphrag/ | Azure sample: https://github.com/Azure-Samples/graphrag-accelerator |
| **Graph RAG Survey (Peng et al.)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | survey | 2024-08 | 2024-09 | — | Paper: https://arxiv.org/abs/2408.08921 | — |
| **GraphRAG Survey (Zhang et al.)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | survey | 2025-01 | 2025-01 | — | Paper: https://arxiv.org/abs/2501.13958 | — |

---

## LLM ↔ Graph Learning

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **GraphICL (Graph In-context Learning Benchmark)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark / node cls / link pred | 2025-01 | 2025-01 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2501.15755 | — |
| **GDL4LLM (Graph-defined Language for LLMs)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred / graph cls | 2025-01 | 2025-01 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2501.11478 | — |
| **TEA-GLM (LLMs as Zero-shot Graph Learners)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred | 2024-08 | 2024-08 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2408.14512 | — |
| **HIGHT (Hierarchical Graph Tokenization)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred | 2024-06 | 2024-06 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2406.14021 | — |
| **GraphTranslator (WWW’24)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred | 2024-05 | 2024-05 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/alibaba/GraphTranslator?style=social) ![last](https://img.shields.io/github/last-commit/alibaba/GraphTranslator) | Paper (ACM): https://dl.acm.org/doi/10.1145/3589334.3645682 · Code: https://github.com/alibaba/GraphTranslator | — |
| **GraphToken / Talk like a Graph** | ![Research](https://img.shields.io/badge/Type-Research-blue) | planning / generation / tokenization | 2024-02 | 2025-10 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/google-research/talk-like-a-graph?style=social) ![last](https://img.shields.io/github/last-commit/google-research/talk-like-a-graph) | Code: https://github.com/google-research/talk-like-a-graph · Paper: https://arxiv.org/abs/2402.05862 | — |
| **LLaGA** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / graph cls | 2024-02 | 2024-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/VITA-Group/LLaGA?style=social) ![last](https://img.shields.io/github/last-commit/VITA-Group/LLaGA) | Paper: https://arxiv.org/abs/2402.08170 · Code: https://github.com/VITA-Group/LLaGA | — |
| **InstructGLM (Language is All a Graph Needs)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred / graph cls | 2023-08 | 2023-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/agiresearch/InstructGLM?style=social) ![last](https://img.shields.io/github/last-commit/agiresearch/InstructGLM) | Paper: https://arxiv.org/abs/2308.07134 · Code: https://github.com/agiresearch/InstructGLM | — |
| **GraphLLM** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / link pred | 2023-10 | 2023-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2310.05845 | — |
| **GraphGPT** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / graph cls | 2023-10 | 2023-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/HKUDS/GraphGPT?style=social) ![last](https://img.shields.io/github/last-commit/HKUDS/GraphGPT) | Code: https://github.com/HKUDS/GraphGPT | — |
| **GPT4Graph** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark / node cls / graph cls | 2023-05 | 2023-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2305.15066 | — |

---

## LLM ↔ Knowledge Graph (KG)

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **Injecting Knowledge Graphs into LLMs** | ![Research](https://img.shields.io/badge/Type-Research-blue) | KG completion / reasoning | 2025-05 | 2025-05 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2505.07554 | — |
| **Graph-Constrained Reasoning (GCR)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | KG reasoning / planning | 2024-10 | 2024-10 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2410.13080 · OpenReview: https://openreview.net/forum?id=Fr7kH2SFq7 | — |
| **Grounding LLM Reasoning with Knowledge Graphs** | ![Research](https://img.shields.io/badge/Type-Research-blue) | KG reasoning | 2024-?? | 2024-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | OpenReview: https://openreview.net/forum?id=2OPS6uhnw6 | — |
| **Evaluating & Enhancing LLMs for KG Reasoning (GPT-4 on KGs)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | KG reasoning | 2023-12 | 2023-12 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2312.11282 | — |
| **TransE** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | KG completion | 2013 | — | — | Paper: https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html | — |
| **DistMult** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | KG completion | 2015 | — | — | Paper: https://arxiv.org/abs/1412.6575 | — |
| **ComplEx** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | KG completion | 2016 | — | — | Paper: https://arxiv.org/abs/1606.06357 | — |
| **RotatE** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | KG completion | 2019 | — | — | Paper: https://arxiv.org/abs/1902.10197 | — |

---

## Graph Storage and Compression (AI-informed dedup)

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **Inference-friendly Graph Compression (IFGC)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | graph compression / GNN inference | 2025-04 | 2025-05 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (arXiv): https://arxiv.org/abs/2504.13034 · Paper (PVLDB): https://www.vldb.org/pvldb/vol18/p3203-fan.pdf | — |
| **Graph Compression for Interpretable GNN Inference at Scale (ExGIS demo)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | graph compression / explainable inference | 2025-?? | 2025-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (PVLDB demo): https://www.vldb.org/pvldb/vol18/p5239-fan.pdf | — |
| **k-HDTDiffCat (Generate & Update Large HDT RDF KGs on Commodity Hardware)** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | RDF KG compression / incremental updates | 2024-04 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/the-qa-company/qEndpoint?style=social) ![last](https://img.shields.io/github/last-commit/the-qa-company/qEndpoint) | Paper (ESWC’24 PDF): https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640460.pdf · Code (qEndpoint): https://github.com/the-qa-company/qEndpoint | — |
| **LSHBloom** | ![Research](https://img.shields.io/badge/Type-Research-blue) | extreme-scale dedup (LSH + Bloom) | 2024-11 | 2024-11 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (arXiv): https://arxiv.org/abs/2411.04257 | LLM dataset curation (as described in paper) |
| **SeqCDC** | ![Research](https://img.shields.io/badge/Type-Research-blue) | CDC / storage dedup acceleration | 2024-12 | 2024-12 | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/UWASL/dedup-bench?style=social) ![last](https://img.shields.io/github/last-commit/UWASL/dedup-bench) | Paper (Middleware’24 PDF): https://cs.uwaterloo.ca/~alkiswan/papers/SeqCDC_Middleware24.pdf · DOI (ACM): https://dl.acm.org/doi/abs/10.1145/3652892.3700766 · Code (DedupBench): https://github.com/UWASL/dedup-bench | — |
| **VectorCDC** | ![Research](https://img.shields.io/badge/Type-Research-blue) | CDC / SIMD acceleration | 2025-02 | 2025-02 | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/UWASL/dedup-bench?style=social) ![last](https://img.shields.io/github/last-commit/UWASL/dedup-bench) | Paper (FAST’25 page): https://www.usenix.org/conference/fast25/presentation/udayashankar · PDF: https://www.usenix.org/system/files/fast25-udayashankar.pdf · Code (DedupBench): https://github.com/UWASL/dedup-bench | — |
| **Accelerating Data Chunking in Deduplication Systems (follow-up to VectorCDC)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | CDC / chunking acceleration | 2025-08 | 2025-08 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (arXiv): https://arxiv.org/abs/2508.05797 | — |
| **Duplicate Detection with GenAI** | ![Research](https://img.shields.io/badge/Type-Research-blue) | semantic dedup / record matching (LLMs) | 2024-06 | 2024-06 | ![Impl](https://img.shields.io/badge/Impl-Maintained-success) ![stars](https://img.shields.io/github/stars/ianormy/genai_duplicate_detection_paper?style=social) ![last](https://img.shields.io/github/last-commit/ianormy/genai_duplicate_detection_paper) | Paper (arXiv): https://arxiv.org/abs/2406.15483 · Code: https://github.com/ianormy/genai_duplicate_detection_paper | — |
| **Match, Compare, or Select? (LLM-based Entity Matching strategies)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | entity matching / semantic dedup (LLMs) | 2025-01 | 2025-01 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (COLING’25 PDF): https://aclanthology.org/2025.coling-main.8.pdf | — |
| **Cross-Dataset Entity Matching with Large & Small LMs (EDBT’25)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | entity matching / cost-quality tradeoffs | 2025-03 | 2025-03 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (EDBT’25 PDF): https://openproceedings.org/2025/conf/edbt/paper-224.pdf | — |
| **LLM-CER (In-context Clustering-based Entity Resolution)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | entity resolution (cluster + LLM) | 2025-06 | 2025-06 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (arXiv HTML): https://arxiv.org/html/2506.02509v1 | — |
| **SemDeDup** | ![Research](https://img.shields.io/badge/Type-Research-blue) | embedding-based semantic dedup | 2023-03 | 2023-03 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper (arXiv): https://arxiv.org/abs/2303.09540 · OpenReview PDF: https://openreview.net/pdf?id=u96ZBg_Shna | Web-scale dataset curation (as described in paper) |
| **datasketch** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | sketches (MinHash/LSH) for dedup | 2016-?? | 2026-?? | ![stars](https://img.shields.io/github/stars/ekzhu/datasketch?style=social) ![last](https://img.shields.io/github/last-commit/ekzhu/datasketch) | Code: https://github.com/ekzhu/datasketch · LSHBloom docs: https://ekzhu.com/datasketch/lshbloom.html | — |

---

## Graph Reasoning Agents and Tool Use

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **GraphAgent-Reasoner** | ![Research](https://img.shields.io/badge/Type-Research-blue) | planning / reasoning | 2024-10 | 2024-10 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2410.05130 | — |
| **GraphAgent (HKUDS)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | planning / node cls / generation | 2024-12 | 2024-12 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/HKUDS/GraphAgent?style=social) ![last](https://img.shields.io/github/last-commit/HKUDS/GraphAgent) | Code: https://github.com/HKUDS/GraphAgent · Paper hub: https://arxiv.org/abs/2412.17029 | — |
| **GraphAgent (ICLR/OpenReview)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | node cls / planning | 2024-?? | 2024-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | OpenReview: https://openreview.net/forum?id=L3jATpVEGv | — |

---

## Benchmarks and Datasets

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **Graph Theory Bench (GT Bench)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark | 2025-?? | 2025-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | OpenReview: https://openreview.net/forum?id=bcGClKY3gQ | — |
| **ProGraph (LLMs Analyze Graphs like Professionals?)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark | 2024-09 | 2024-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/BUPT-GAMMA/ProGraph?style=social) ![last](https://img.shields.io/github/last-commit/BUPT-GAMMA/ProGraph) | Paper: https://arxiv.org/abs/2409.19667 · Code: https://github.com/BUPT-GAMMA/ProGraph | HF models/datasets: https://huggingface.co/lixin4sky/ProGraph |
| **GraphPile (CPT corpus for graph problem reasoning)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | dataset / CPT | 2025-?? | 2026-01-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | OpenReview: https://openreview.net/forum?id=6vMRcaYbU7 | — |
| **GPT4Graph benchmark** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark | 2023-05 | 2023-?? | — | Paper: https://arxiv.org/abs/2305.15066 | — |
| **OGB (Open Graph Benchmark)** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | benchmark | 2020 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/snap-stanford/ogb?style=social) ![last](https://img.shields.io/github/last-commit/snap-stanford/ogb) | Site: https://ogb.stanford.edu/ · Code: https://github.com/snap-stanford/ogb | — |

---

## Graph Generation (LLM / Agents / Diffusion)

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **GraphAgent-Generator (GAG)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | generation | 2024-10 | 2024-10 | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | Paper: https://arxiv.org/abs/2410.09824 | — |
| **Graph DiT** | ![Research](https://img.shields.io/badge/Type-Research-blue) | generation | 2024-01 | 2024-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/liugangcode/Graph-DiT?style=social) ![last](https://img.shields.io/github/last-commit/liugangcode/Graph-DiT) | Paper: https://arxiv.org/abs/2401.13858 · Code: https://github.com/liugangcode/Graph-DiT | — |
| **GraphRNN** | ![Research](https://img.shields.io/badge/Type-Research-blue) | generation | 2018 | — | — | Paper: https://arxiv.org/abs/1802.08773 | — |
| **Junction Tree VAE** | ![Research](https://img.shields.io/badge/Type-Research-blue) | generation | 2018 | — | — | Paper: https://arxiv.org/abs/1802.04364 | — |

---

## Neural Algorithmic Graph Reasoning

| Name | Type | Task | Publication date | Update date | Badges | Links | Used in |
|---|---|---|---:|---:|---|---|---|
| **Think in Graphs (OpenReview)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark / reasoning | 2025-?? | 2025-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) | OpenReview: https://openreview.net/forum?id=DczVG12sdJ | — |
| **Neural Algorithmic Reasoning (DeepMind)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | reasoning | 2021 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/deepmind/neural-algorithmic-reasoning?style=social) ![last](https://img.shields.io/github/last-commit/deepmind/neural-algorithmic-reasoning) | Paper: https://arxiv.org/abs/2105.02761 · Code: https://github.com/deepmind/neural-algorithmic-reasoning | — |
| **CLRS Benchmark (DeepMind)** | ![Research](https://img.shields.io/badge/Type-Research-blue) | benchmark | 2022 | 2026-?? | ![Impl](https://img.shields.io/badge/Impl-Unknown-lightgrey) ![stars](https://img.shields.io/github/stars/deepmind/clrs?style=social) ![last](https://img.shields.io/github/last-commit/deepmind/clrs) | Paper: https://arxiv.org/abs/2205.15659 · Code: https://github.com/deepmind/clrs | — |

---

## Production Graph ML Systems

| Name | Type | Task | Publication date | Update date | Links | Used in |
|---|---|---|---:|---:|---|---|
| **AliGraph** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | systems | 2019 | — | Paper: https://arxiv.org/abs/1902.08730 | Alibaba (as described in paper) |
| **PinSage** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | recommendation / link pred | 2018 | — | Paper: https://arxiv.org/abs/1806.01973 | Pinterest (as described in paper) |

---

## Libraries

| Name | Type | Focus | Update date | Badges | Links |
|---|---|---|---:|---|---|
| **PyTorch Geometric** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | GNN framework | 2026-?? | ![stars](https://img.shields.io/github/stars/pyg-team/pytorch_geometric?style=social) ![last](https://img.shields.io/github/last-commit/pyg-team/pytorch_geometric) | https://github.com/pyg-team/pytorch_geometric |
| **DGL** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | GNN framework | 2026-?? | ![stars](https://img.shields.io/github/stars/dmlc/dgl?style=social) ![last](https://img.shields.io/github/last-commit/dmlc/dgl) | https://github.com/dmlc/dgl |
| **NetworkX** | ![Production](https://img.shields.io/badge/Type-Production-brightgreen) | classic graph algorithms | 2026-?? | ![stars](https://img.shields.io/github/stars/networkx/networkx?style=social) ![last](https://img.shields.io/github/last-commit/networkx/networkx) | https://github.com/networkx/networkx |

---

## Meta Lists and Surveys

| Name | What it’s for | Update date | Links |
|---|---|---:|---|
| **Awesome-LLMs-in-Graph-tasks** | survey-backed paper catalogue (LLM4Graph) | 2025-?? | https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks |
| **Awesome-LLM-KG** | LLM↔KG resources (KG-enhanced LLMs + LLM-augmented KGs) | 2026-?? | https://github.com/RManLuo/Awesome-LLM-KG |
| **HKUDS Awesome-LLM4Graph-Papers** | curated buckets by paradigm | 2026-?? | https://github.com/HKUDS/Awesome-LLM4Graph-Papers |
| **GraphRAG Survey (Peng et al.)** | GraphRAG workflow taxonomy | 2024-09 | https://arxiv.org/abs/2408.08921 |
| **GraphRAG Survey (Zhang et al.)** | systematic GraphRAG analysis | 2025-01 | https://arxiv.org/abs/2501.13958 |
| **Awesome Language Model on Graphs** | broader “LM on graphs” paper list | 2026-?? | https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs |

---

## Contributing

PRs welcome. Please include for each entry:
- Paper link (arXiv/OpenReview/venue)
- Code link (if any)
- **Publication date** and **Update date**
- Type (Production/Research)
- Task tags
- “Used in” only with a concrete public reference (paper statement, docs, or engineering blog)

---

## Citation

If you use this list in academic work, cite the relevant surveys/meta-lists above plus the specific papers referenced per section.

---

## License

Recommended for this list: **CC0-1.0** (or MIT).  
Linked papers and repositories retain their original licenses.

---

## Inspirations

This list is inspired by:
- https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks
- https://github.com/RManLuo/Awesome-LLM-KG
- https://github.com/HKUDS/Awesome-LLM4Graph-Papers
- (format inspiration) https://github.com/BunnySoCrazy/Awesome-Neural-CAD
