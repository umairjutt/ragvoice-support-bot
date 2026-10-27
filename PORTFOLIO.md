# Engineering guide

Experimental local RAG assistant with hybrid retrieval, reranking, streamed answers and voice integration.

## Code tour

- [backend/app/retrieval.py](backend/app/retrieval.py)
- [backend/app/chat.py](backend/app/chat.py)
- [backend/app/eval.py](backend/app/eval.py)
- [backend/tests](backend/tests)

## Scope and verification

Experimental local RAG application. Dependency installation, authenticated tenant access, upload boundaries and voice-model provisioning need further work before multi-user deployment.

[Current CI runs](https://github.com/umairjutt/ragvoice-support-bot/actions). A workflow definition or build result does not establish that all runtime scenarios pass. The README contains development commands; unfinished setup and behavior need completion before production deployment.

## Maintainer

[Umair Jutt](https://github.com/umairjutt). Frameworks and dependencies retain their own ownership and licensing.
