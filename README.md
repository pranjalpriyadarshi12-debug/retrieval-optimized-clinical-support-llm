# Clinical Risk Factor & Recommendation System  
### Retrieval-Optimized LLM Pipeline for Decision Support

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
