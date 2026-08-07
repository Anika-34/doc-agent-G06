# Per-stage design choices (A2 deliverable). Fill every cell.
| Stage | Problem statement | Data | Model | Methods | Design | Development | Deployment | MLOps |
|---|---|---|---|---|---|---|---|---|
| 0 Frame | Build a grounded scanned-page QA agent for soil science and crop-rotation decisions from historical Bangla manuals. | 3 Internet Archive scanned books, 776 pages, non-English Bangla script with degraded scans and old orthography. | Stage 0 is model-agnostic; establishes constraints and acceptance targets for downstream model choices. | Commit to evidence-grounded answers with abstention on weak retrieval evidence; optimize reliability over recall. | Fixed pipeline contracts and stage ordering from starter repo; document-by-document split to prevent leakage. | Encode decisions in `configs/task.yaml`, provenance in `data/provenance.md`, and this per-stage table as A2 baseline. | Deployable target is API-served doc-agent with citations, but Stage 0 only sets the operational objective and constraints. | Reproducible seed/config driven runs; track target metric: answer reliability >= 0.90 under noisy OCR conditions. |
| 1 Ingest+Enhance |  |  |  |  |  |  |  |  |
| 2 Layout |  |  |  |  |  |  |  |  |
| 3 OCR |  |  |  |  |  |  |  |  |
| 4 Index |  |  |  |  |  |  |  |  |
| 5 Retrieval |  |  |  |  |  |  |  |  |
| 6 Agent |  |  |  |  |  |  |  |  |
| 7 RL/RLVR |  |  |  |  |  |  |  |  |
| 8 Serving |  |  |  |  |  |  |  |  |
| 9 Eval |  |  |  |  |  |  |  |  |
