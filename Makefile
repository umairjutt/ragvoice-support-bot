.PHONY: up down logs ingest eval eval-ci test lint pull-models

up: ; docker compose up -d
down: ; docker compose down -v
logs: ; docker compose logs -f api

pull-models:
	docker compose exec ollama ollama pull llama3.1:8b
	docker compose exec ollama ollama pull nomic-embed-text

ingest:
	docker compose exec api python -m app.ingest /data/sample_docs default

eval:
	docker compose exec api python -m app.eval

eval-ci:
	cd backend && python -m app.eval --ci

test:
	cd backend && pytest

lint:
	cd backend && ruff check app tests && mypy app
