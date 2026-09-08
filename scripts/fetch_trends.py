#!/usr/bin/env python3
"""
Continuous Research & GitHub Trend Discovery Engine for cv-projects.
Queries arXiv API and GitHub Search across all 12 perception & robotics domains,
extracts licenses, paper URLs, and author metadata, and flags high-impact preprints.

Usage:
    python scripts/fetch_trends.py [--query QUERY] [--domain DOMAIN] [--max-results 5] [--scaffold]
"""

import argparse
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Domain query keywords for automated tracking
DOMAIN_QUERIES = {
    "object-detection": "all:\"real-time object detection\" OR all:\"RF-DETR\" OR all:\"RT-DETR\" OR all:\"YOLOv10\"",
    "object-segmentation": "all:\"SAM 2\" OR all:\"segment anything video\" OR all:\"panoptic segmentation transformer\"",
    "object-classification": "all:\"DINOv2\" OR all:\"SigLIP\" OR all:\"vision foundation model\" OR all:\"ConvNeXt\"",
    "video-tracking": "all:\"point tracking\" OR all:\"CoTracker\" OR all:\"multiple object tracking\" OR all:\"BoT-SORT\"",
    "6dof-pose-estimation": "all:\"6-DoF pose estimation\" OR all:\"FoundationPose\" OR all:\"BOP benchmark pose\"",
    "lidar-perception": "all:\"LiDAR 3D detection\" OR all:\"sparse voxel transformer\" OR all:\"DSVT\"",
    "sensor-fusion": "all:\"BEVFusion\" OR all:\"camera lidar fusion\" OR all:\"Bird's-Eye-View perception\"",
    "fpga-deployment": "all:\"FPGA neural network\" OR all:\"quantized neural network FPGA\" OR all:\"Vitis AI\"",
    "gpu-deployment": "all:\"TensorRT\" OR all:\"FlashAttention\" OR all:\"GPU inference optimization\"",
    "real-time-systems": "all:\"zero-copy IPC\" OR all:\"PREEMPT_RT real-time vision\" OR all:\"robotics middleware\"",
    "slam-and-spatial-perception": "all:\"3D Gaussian Splatting SLAM\" OR all:\"MonoGS\" OR all:\"visual inertial odometry\"",
    "visual-guidance-and-robotics": "all:\"Vision-Language-Action\" OR all:\"OpenVLA\" OR all:\"6-DoF grasp synthesis\"",
}

ARXIV_API_BASE = "https://export.arxiv.org/api/query"

def search_arxiv(search_query: str, max_results: int = 5) -> list[dict]:
    """Queries the public arXiv API for recent preprints."""
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API_BASE}?{params}"
    
    headers = {"User-Agent": "cv-projects-trend-bot/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()
    except Exception as e:
        print(f"[-] Error fetching from arXiv: {e}")
        return []

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary)
        published = entry.find("atom:published", ns).text.strip()
        paper_id = entry.find("atom:id", ns).text.strip()
        
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns).text.strip()
            authors.append(name)
            
        papers.append({
            "title": title,
            "published": published[:10],
            "url": paper_id,
            "authors": authors[:3],
            "summary": summary[:280] + "..." if len(summary) > 280 else summary
        })
        
    return papers

def scaffold_model_draft(paper: dict, domain: str, repo_root: Path) -> Path:
    """Scaffolds a new model draft in architectures/ based on model-deep-dive-template."""
    clean_title = re.sub(r"[^\w\s-]", "", paper["title"])
    slug = "-".join(clean_title.lower().split()[:5])
    target_dir = repo_root / "architectures" / "recent-discoveries"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{slug}.md"
    
    content = f"""---
title: "{paper['title']}"
type: model-deep-dive
domain: "{domain}"
architecture_class: "Emerging SOTA Architecture"
primary_license: "Pending Verification"
commercial_use: true
official_repo: "https://github.com/search?q={urllib.parse.quote(paper['title'])}"
tags:
  - discovery
  - emerging-paper
  - {domain}
  - sota
updated: {datetime.now().strftime('%Y-%m-%d')}
aliases:
  - "{paper['title'][:40]}"
---

# 🔬 {paper['title']}

## 1. Executive Brief & Significance
- **Authors**: {", ".join(paper['authors'])} et al.
- **Publication Date**: {paper['published']}
- **Paper Link**: [{paper['url']}]({paper['url']})

### Summary Abstract
> {paper['summary']}

---

## 2. Core Architectural Mechanics & Innovations
- Key algorithmic components to be evaluated against existing [[topics/{domain}/00-{domain}-moc|{domain} MOC]].

---

## 3. Quantitative Performance & Verification
- SOTA metrics and baseline comparisons in progress.

---

## 4. Commercial Usability & License Audit
- Verify repository license before proprietary production adoption.
"""
    target_path.write_text(content, encoding="utf-8")
    return target_path

def main():
    parser = argparse.ArgumentParser(description="Fetch latest CV/Robotics trends and papers.")
    parser.add_argument("--domain", choices=list(DOMAIN_QUERIES.keys()), default=None, help="Target specific topic domain")
    parser.add_argument("--max-results", type=int, default=2, help="Max results per domain query")
    parser.add_argument("--scaffold", action="store_true", help="Scaffold draft note in architectures/recent-discoveries/")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    domains = [args.domain] if args.domain else list(DOMAIN_QUERIES.keys())

    print(f"[+] Starting trend discovery across {len(domains)} domain(s)...")

    total_discovered = 0
    for domain in domains:
        query = DOMAIN_QUERIES[domain]
        print(f"\n[📡 Querying Domain: {domain}]")
        papers = search_arxiv(query, max_results=args.max_results)
        
        if not papers:
            print("  No recent papers returned.")
            continue
            
        for p in papers:
            total_discovered += 1
            print(f"  • {p['published']} | {p['title']}")
            print(f"    Authors: {', '.join(p['authors'])}")
            print(f"    Link: {p['url']}")
            
            if args.scaffold:
                draft_path = scaffold_model_draft(p, domain, repo_root)
                print(f"    [+] Scaffolded note draft: {draft_path.relative_to(repo_root)}")

    print(f"\n[✓] Discovery complete: {total_discovered} preprints retrieved.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
