#!/usr/bin/env bun
/**
 * Strict Mermaid Diagram Validator for cv-projects repository.
 * Parses all ```mermaid blocks across markdown files using official Mermaid JS parser.
 * Exits with code 0 on success, or 1 with diagnostic details on any parse error.
 */

import fs from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';

// Initialize headless DOM environment required by Mermaid in non-browser runtimes
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.DOMPurify = (await import('dompurify')).default(dom.window);

const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, suppressErrorRendering: true });

function findMarkdownFiles(dir) {
  let results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (
        entry.name !== '.git' &&
        entry.name !== 'node_modules' &&
        entry.name !== '.obsidian' &&
        entry.name !== '.venv'
      ) {
        results = results.concat(findMarkdownFiles(fullPath));
      }
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      results.push(fullPath);
    }
  }
  return results;
}

async function validateAllMermaid() {
  const files = findMarkdownFiles('.');
  const mermaidRegex = /^```mermaid\s*\n([\s\S]*?)```/gm;

  let totalDiagrams = 0;
  let passedDiagrams = 0;
  let errors = [];

  for (const file of files) {
    const content = fs.readFileSync(file, 'utf-8');
    let match;
    let index = 0;
    while ((match = mermaidRegex.exec(content)) !== null) {
      totalDiagrams++;
      index++;
      const code = match[1].trim();
      try {
        await mermaid.parse(code);
        passedDiagrams++;
      } catch (err) {
        errors.push({
          file,
          index,
          error: err.message?.split('\n')[0] || String(err),
          snippet: code.split('\n').slice(0, 5).join('\n')
        });
      }
    }
  }

  console.log(`[+] Scanned ${files.length} markdown files, found ${totalDiagrams} Mermaid diagrams.`);

  if (errors.length > 0) {
    console.error(`[-] Validation FAILED: ${errors.length} diagram(s) failed parsing:`);
    for (const e of errors) {
      console.error(`    - ${e.file} (diagram #${e.index}): ${e.error}`);
      console.error(`      Snippet: ${e.snippet.replace(/\n/g, ' ')}`);
    }
    process.exit(1);
  } else {
    console.log(`[+] Validation PASSED: All ${totalDiagrams} Mermaid diagrams parsed successfully.`);
    process.exit(0);
  }
}

await validateAllMermaid();
