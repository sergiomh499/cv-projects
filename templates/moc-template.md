---
id: "{{id}}"
title: "{{title}}"
type: MOC
domain: "{{domain}}"
tags:
  - moc
  - computer-vision
  - "{{domain_tag}}"
status: evergreen
updated: YYYY-MM-DD
aliases:
  - "{{title}}"
  - "{{domain}} Hub"
---

# 🗺️ {{title}} (Map of Content)

> [!NOTE] Domain Overview & Scope
> A high-level executive synthesis of the perception domain, hardware constraints, and primary failure modes. This MOC is the canonical entry point; all sub-notes should link back here.

## 🧭 Navigation & Knowledge Graph

- **Historical Evolution & Paradigms**: [[{{domain_path}}/evolution/00-historical-lineage-and-paradigms|Historical Lineage & Architectural Evolution]]
- **Production Implementation Playbook**: [[{{domain_path}}/playbooks/00-production-pipeline-and-workarounds|Production Pipeline, Engineering Traps & Workarounds]]

### 🔬 In-Depth Model & Architecture Notes

| Model / System | Architecture Class | Primary Innovation | License | Note Link |
| :--- | :--- | :--- | :---: | :---: |
| **Model A** | Transformer / CNN | ... | Apache-2.0 | [[{{domain_path}}/models/model-a\|Model A Deep-Dive]] |
| **Model B** | Foundation Model | ... | MIT | [[{{domain_path}}/models/model-b\|Model B Deep-Dive]] |

---

## 📊 Standardized SOTA Benchmark Comparison

| Model Variant | Key Dataset | Metric | Latency (FP16 ms) | Hardware Profile | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ... | ... | ... | ... | ... | ... |

> [!WARNING] Benchmark Gotchas
> Latency figures are hardware-specific. Always re-profile on your target device. NMS overhead is frequently omitted from published numbers — include it in your own measurements.

---

## ⚖️ Commercial Usability & License Matrix

- **Commercial Permissive (Apache-2.0 / MIT / BSD)**: ...
- **Copyleft Warning (AGPL / GPL)**: ...
- **Research-Only / Non-Commercial**: ...

> [!TIP] License Audit
> AGPL-3.0 requires releasing network-facing service modifications. If shipping a SaaS product, prefer Apache-2.0 or MIT alternatives or obtain a commercial license before integrating.

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(domain, "{{domain}}")
SORT file.name ASC
```
