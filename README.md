# Clinical Risk Factor & Recommendation System  
### Retrieval-Optimized LLM Pipeline for Decision Support

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/LLM-Azure%20OpenAI-purple"/>
  <img src="https://img.shields.io/badge/Architecture-RAG-green"/>
  <img src="https://img.shields.io/badge/Focus-Prompt%20Optimization-orange"/>
</p>
---

## Overview

A **retrieval-optimized LLM system** that processes patient data to:

- Extract **clinically relevant risk factors**
- Recommend the most appropriate **clinical group**
- Reduce **LLM cost, latency, and prompt size**

> Instead of sending the full dataset to the model, this system retrieves only the **most relevant candidates** before inference.

---

## Key Impact

| Metric | Baseline | Optimized | Improvement |
|------  |--------  |---------- |------------ |
| Prompt size | 677K chars | 122K | 🔻 **-82%** |
| Risk factor entries | 517 | 40 | 🔻 **-92%** |
| Clinical groups evaluated | 43 | 8 | 🔻 **-81%** |

## Architecture
```mermaid
flowchart TD
    A[Patient Data + Notes] --> B[Text Extraction]
    B --> C[Risk Factor Retrieval]
    C --> D[Top-K Candidates]
    D --> E[LLM Risk Factor Extraction]
    E --> F[Normalized Risk Factors]
    F --> G[Clinical Group Retrieval]
    G --> H[Top-K Groups]
    H --> I[LLM Recommendation]
    I --> J[Final JSON Output]
