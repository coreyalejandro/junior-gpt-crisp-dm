import json
from pathlib import Path
from jsonschema import Draft202012Validator

root = Path(__file__).resolve().parents[1]
schema_path = root / 'contracts' / 'events.schema.json'
raw = schema_path.read_text(encoding='utf-8')
Draft202012Validator.check_schema(json.loads(raw))
print('OK: events.schema.json is valid')
