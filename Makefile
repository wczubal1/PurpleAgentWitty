SHELL := /bin/bash

PURPLE_HOST ?= 127.0.0.1
PURPLE_PORT ?= 9010

.PHONY: run
run:
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY" >&2; exit 1)
	uv run src/server.py --host $(PURPLE_HOST) --port $(PURPLE_PORT)

.PHONY: docker-build
docker-build:
	docker build -t purple-agent .

.PHONY: docker-run
docker-run:
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY" >&2; exit 1)
	docker run -p $(PURPLE_PORT):9010 -e OPENAI_API_KEY="$$OPENAI_API_KEY" purple-agent --host 0.0.0.0 --port 9010
