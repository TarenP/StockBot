"""
One-time cleanup of the document store.

1. Removes 10-K and 10-Q documents (too long, cause timeouts)
2. Re-cleans HTML entities in remaining 8-K documents
3. Removes documents that fail the _is_useful_for_llm filter

Run once:
    python clean_document_store.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE_DIR = Path("broker/state/document_store")
KEEP_FORMS = {"8_k", "earnings_news", "8-k"}  # only keep these source types


def clean():
    import html as _html
    import re

    def strip_html(text):
        text = _html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_useful(text):
        if len(text) < 200:
            return False
        words = text.split()
        if len(words) < 50:
            return False
        numeric = sum(1 for w in words[:200] if w.replace(".", "").replace(",", "").replace("$", "").replace("-", "").isdigit())
        if numeric / min(len(words), 200) > 0.5:
            return False
        return True

    files = sorted(STORE_DIR.glob("*.json"))
    print(f"Found {len(files)} documents in store")

    removed = 0
    cleaned = 0
    kept = 0

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            removed += 1
            continue

        source_type = str(doc.get("source_type", "")).lower()

        # Remove non-8K documents
        if source_type not in KEEP_FORMS:
            path.unlink(missing_ok=True)
            removed += 1
            continue

        # Re-clean HTML entities
        text = doc.get("text", "")
        clean_text = strip_html(text)[:8000]

        # Remove if not useful after cleaning
        if not is_useful(clean_text):
            path.unlink(missing_ok=True)
            removed += 1
            continue

        # Update with cleaned text if changed
        if clean_text != text:
            doc["text"] = clean_text
            path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
            cleaned += 1
        else:
            kept += 1

    print(f"Removed: {removed} (10-K/10-Q/XBRL/unusable)")
    print(f"Cleaned: {cleaned} (HTML entities stripped)")
    print(f"Kept:    {kept} (already clean)")
    print(f"Remaining: {removed + cleaned + kept - removed} documents ready for LLM parsing")


if __name__ == "__main__":
    clean()
