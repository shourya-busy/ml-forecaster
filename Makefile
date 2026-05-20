SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help build up down logs ps test lint seed migrate shell-api shell-worker reset gpu demo

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all images
	$(COMPOSE) build

up: ## Start full stack
	$(COMPOSE) up -d

down: ## Stop stack (keeps volumes)
	$(COMPOSE) down

reset: ## Stop stack + drop volumes
	$(COMPOSE) down -v

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

migrate: ## Run alembic migrations
	$(COMPOSE) run --rm migrate

seed: ## Seed synthetic series into the mock Prometheus or DB
	$(COMPOSE) exec api forecaster-seed --instances fake-1,fake-2 --days 30

test: ## Run unit tests inside the worker image
	$(COMPOSE) run --rm worker python -m pytest -q

lint:
	ruff check .

shell-api:
	$(COMPOSE) exec api bash

shell-worker:
	$(COMPOSE) exec worker bash

gpu: ## Start with GPU override
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d

demo: ## Start full stack with demo Prometheus + Grafana
	$(COMPOSE) --profile demo up -d

preflight: ## Release-readiness checks (services, /metrics, end-to-end run)
	bash scripts/preflight.sh
