.PHONY: up down logs ingest eval test pull-models

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
