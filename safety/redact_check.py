import re
import sys
from pathlib import Path

PII_PATTERNS = [
    re.compile(r"[\w.-]+@[\w.-]+"),
    re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"),
]

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
violations = 0
for path in root.rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.ndjson', '.txt', '.md'}:
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for pat in PII_PATTERNS:
            if pat.search(text):
                print(f"Potential PII in {path}")
                violations += 1

if violations:
    sys.exit(1)
print('No PII detected')
