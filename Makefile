.PHONY: install run test eval clean validate

install:
	python3 -m venv .venv && ./.venv/bin/pip install --upgrade pip && ./.venv/bin/pip install -r requirements.txt

run:
	./.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

test:
	./.venv/bin/python -m pytest tests/ -v

eval:
	./.venv/bin/python scripts/eval/evaluator.py

validate:
	./.venv/bin/python scripts/validate_schemas.py

clean:
	rm -rf .venv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

dev:
	./.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 --log-level debug

prod:
	./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4

health:
	curl -f http://localhost:8001/health || echo "Service not running"

tools:
	curl -s http://localhost:8001/tools | python3 -m json.tool

agents:
	curl -s http://localhost:8001/agents | python3 -m json.tool

demo:
	@echo "Starting demo session..."
	curl -X POST http://localhost:8001/session/start \
		-H "Content-Type: application/json" \
		-H "Accept: text/event-stream" \
		-d '{"task": "What is the capital of France?", "agent": "analyst"}' \
		--no-buffer

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  run        - Run development server"
	@echo "  test       - Run tests"
	@echo "  eval       - Run evaluation suite"
	@echo "  validate   - Validate schemas"
	@echo "  clean      - Clean up files"
	@echo "  dev        - Run with debug logging"
	@echo "  prod       - Run production server"
	@echo "  health     - Check service health"
	@echo "  tools      - List available tools"
	@echo "  agents     - List available agents"
	@echo "  demo       - Run demo session"
	@echo "  help       - Show this help"
