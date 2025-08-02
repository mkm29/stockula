# Makefile for Stockula Docker operations
# Provides convenient commands for building and running Docker containers

.PHONY: help build build-all test clean dev jupyter cli run-example lint format check-deps docs docs-serve docs-build docs-deploy

# Default target
help: ## Show this help message
	@echo "Stockula Docker & Documentation Makefile"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build production image
	docker build --target production -t stockula:latest .

build-dev: ## Build development image
	docker build --target development -t stockula:dev .

build-cli: ## Build CLI image
	docker build --target cli -t stockula:cli .

build-jupyter: ## Build Jupyter image
	docker build --target jupyter -t stockula:jupyter .

build-test: ## Build test image
	docker build --target test -t stockula:test .

build-all: ## Build all images
	docker build --target production -t stockula:latest .
	docker build --target development -t stockula:dev .
	docker build --target cli -t stockula:cli .
	docker build --target jupyter -t stockula:jupyter .
	docker build --target test -t stockula:test .

# Development targets
dev: ## Start development environment with Jupyter
	docker-compose up stockula-dev

dev-shell: ## Start development shell
	docker-compose run --rm stockula-dev /bin/bash

jupyter: ## Start Jupyter Lab environment
	docker-compose up stockula-jupyter

cli: ## Start CLI environment
	docker-compose run --rm stockula-cli

cli-shell: ## Start CLI shell
	docker-compose run --rm stockula-cli /bin/bash

# Testing targets
test: ## Run all tests in Docker
	docker-compose up stockula-test

test-unit: ## Run unit tests only
	docker run --rm -v $(PWD):/app stockula:test uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	docker run --rm -v $(PWD):/app stockula:test uv run pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	docker run --rm -v $(PWD):/app stockula:test uv run pytest tests/ --cov=src/stockula --cov-report=html

lint: ## Run linting checks (consistent with CI)
	uv run lint

lint-docker: ## Run linting in Docker
	docker run --rm -v $(PWD):/app stockula:test uv run ruff check src/ tests/

format: ## Format code using ruff
	uv run ruff format src tests

format-check: ## Check code formatting
	uv run ruff format --check src tests

format-docker: ## Format code using ruff in Docker
	docker run --rm -v $(PWD):/app stockula:test uv run ruff format src/ tests/

format-check-docker: ## Check code formatting in Docker
	docker run --rm -v $(PWD):/app stockula:test uv run ruff format --check src/ tests/

# Example running targets
run-example-dynamic: ## Run dynamic rates example
	docker run --rm \
		-v stockula-data:/app/data \
		-v stockula-results:/app/results \
		stockula:cli python examples/automatic_dynamic_rates_example.py

run-example-treasury: ## Run treasury rates example
	docker run --rm \
		-v stockula-data:/app/data \
		-v stockula-results:/app/results \
		stockula:cli python examples/treasury_rate_example.py

run-example-sharpe: ## Run dynamic Sharpe ratio example
	docker run --rm \
		-v stockula-data:/app/data \
		-v stockula-results:/app/results \
		stockula:cli python examples/dynamic_sharpe_example.py

# Utility targets
clean: ## Clean up Docker images and containers
	docker-compose down --remove-orphans
	docker system prune -f

clean-all: ## Clean up everything including volumes
	docker-compose down --remove-orphans --volumes
	docker system prune -af

logs: ## Show logs from development container
	docker-compose logs -f stockula-dev

logs-jupyter: ## Show logs from Jupyter container
	docker-compose logs -f stockula-jupyter

ps: ## Show running containers
	docker-compose ps

# Volume management
backup-data: ## Backup data volume
	docker run --rm \
		-v stockula-data:/data \
		-v $(PWD)/backup:/backup \
		alpine tar czf /backup/stockula-data-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .

restore-data: ## Restore data volume (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Usage: make restore-data BACKUP_FILE=path/to/backup.tar.gz"; exit 1; fi
	docker run --rm \
		-v stockula-data:/data \
		-v $(PWD)/backup:/backup \
		alpine tar xzf /backup/$(BACKUP_FILE) -C /data

list-volumes: ## List Docker volumes
	docker volume ls | grep stockula

inspect-data-volume: ## Inspect data volume
	docker volume inspect stockula-data

inspect-results-volume: ## Inspect results volume
	docker volume inspect stockula-results

# Dependencies and security
check-deps: ## Check for dependency vulnerabilities
	docker run --rm -v $(PWD):/app stockula:test uv run pip-audit

size: ## Show image sizes
	@echo "Docker image sizes:"
	@docker images | grep stockula | awk '{print $$1":"$$2" - "$$7$$8}'

# Docker buildx for multi-platform builds
buildx-setup: ## Setup buildx for multi-platform builds
	docker buildx create --use --name stockula-builder

buildx-build: ## Build multi-platform images
	docker buildx build --platform linux/amd64,linux/arm64 --target production -t stockula:latest .

# Production deployment helpers
tag-latest: ## Tag latest image for registry
	docker tag stockula:latest $(REGISTRY)/stockula:latest

tag-version: ## Tag version (requires VERSION variable)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag-version VERSION=v1.0.0"; exit 1; fi
	docker tag stockula:latest $(REGISTRY)/stockula:$(VERSION)

push-latest: ## Push latest to registry
	docker push $(REGISTRY)/stockula:latest

push-version: ## Push version to registry (requires VERSION variable)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make push-version VERSION=v1.0.0"; exit 1; fi
	docker push $(REGISTRY)/stockula:$(VERSION)

# Health checks
health-check: ## Check container health
	@docker inspect --format='{{.State.Health.Status}}' stockula-dev 2>/dev/null || echo "Container not running"

# Quick development workflow
quick-test: build-test test ## Quick build and test cycle

quick-dev: build-dev dev ## Quick build and start development

# Documentation targets
docs: docs-build ## Start MkDocs development server
	uv run mkdocs serve

docs-serve: docs-build ## Start MkDocs development server (alias for docs)
	uv run mkdocs serve

docs-build: ## Build MkDocs site
	uv run mkdocs build

docs-deploy: ## Deploy MkDocs site to GitHub Pages
	uv run mkdocs gh-deploy

docs-clean: ## Clean MkDocs build directory
	rm -rf site/

# Variables
REGISTRY ?= your-registry.com
PWD := $(shell pwd)
