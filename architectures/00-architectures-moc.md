---
title: "Central Architecture Vault & Taxonomy Hub MOC"
type: MOC
domain: Computer Vision & AI Architectures
architecture_class: "Taxonomy Hub"
primary_license: "Apache-2.0 / MIT"
tags:
  - moc
  - architecture
  - foundation-models
  - real-time
  - taxonomy
status: evergreen
updated: 2026-09-08
aliases:
  - Architecture Hub
  - Architecture Vault
  - Architecture Taxonomy
---

# 🏛️ Central Architecture Vault & Taxonomy Hub MOC

> **Canonical Hub**: Please refer to [[architectures/README|Central Architecture Vault & Taxonomy Hub]] for the complete high-level taxonomy, component-by-component breakdown tables, and benchmark profiles.

```dataview
TABLE architecture_class AS "Class", primary_license AS "License", updated AS "Updated"
FROM "architectures"
WHERE file.name != "README" AND file.name != "00-architectures-moc"
SORT file.name ASC
```
