"""Measures the real page of every heading/caption in the Word-exported PDF of the report and stores it in mining_cache/report_pages.json
(used by 14_build_project_report.py as the cached value of the page-number fields)."""
import json, re, sys
from pathlib import Path
import pypdf
ROOT = Path(__file__).resolve().parent.parent
pdf = sys.argv[1]; texts = json.load(open(ROOT / "mining_cache" / "report_bm_text.json"))
pages = [re.sub(r"[ \t]+", " ", p.extract_text() or "") for p in pypdf.PdfReader(pdf).pages]
def find(pat, start=0, cond=None):
    for i in range(start, len(pages)):
        if re.search(pat, pages[i]) and (cond is None or cond(pages[i])): return i + 1
    return None
out = {"abstract": find(r"ABSTRACT\s*\nFlight delays"), "lof": find(r"LIST OF FIGURES\s*\n\s*\nSR"), "loa": find(r"LIST OF ABBREVIATIONS"), "lot": find(r"LIST OF TABLES")}
out["loa"] = find(r"LIST OF ABBREVIATIONS", cond=lambda t: "SR.NO" in t or "ABBREVIATION" in t and "TABLE OF CONTENTS" not in t); out["lot"] = find(r"LIST OF TABLES", cond=lambda t: "TABLE OF CONTENTS" not in t)
body0 = find(r"CHAPTER 1\s*\n\s*INTRODUCTION") or 1
for name, t in texts.items():
    if name.startswith("ch"): out[name] = find(rf"{re.escape(t)}\s*\n", body0 - 1)
    else: out[name] = find(re.escape(t), body0 - 1)
json.dump(out, open(ROOT / "mining_cache" / "report_pages.json", "w"), indent=0); print({k: v for k, v in out.items() if v is None} or "all headings found", "| pages", len(pages))
print({k: out[k] for k in ["abstract", "lof", "loa", "lot", "ch1", "ch4", "ch5", "ch6", "ch7"]})
