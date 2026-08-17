.PHONY: help install install-all install-dev test test-unit test-integration test-verbose test-coverage lint format type-check check-all build clean release publish dev-setup shell

# Project settings
PROJECT_NAME := memtuner
VERSION := 0.0.1
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := benchmark
TEST_DIR := tests

# Help target
help:
	@echo "$(PROJECT_NAME) v$(VERSION) - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install           Install production dependencies"
	@echo "  make install-all       Install with all optional dependencies"
	@echo "  make install-dev       Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run all tests"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-verbose      Run tests with verbose output"
	@echo "  make test-coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              Check code with ruff"
	@echo "  make format            Format code with ruff"
	@echo "  make type-check        Type check with mypy"
	@echo "  make check-all         Run all checks (lint, type, test)"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build             Build distribution packages"
	@echo "  make clean             Clean build artifacts"
	@echo "  make release           Create release"
	@echo "  make publish           Publish to PyPI"
	@echo ""

# Installation targets
install:
	$(PIP) install -e .

install-all:
	$(PIP) install -e ".[all]"

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

# Testing targets
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

test-unit:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m unit --tb=short

test-integration:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m integration --tb=short

test-verbose:
	$(PYTHON) -m pytest $(TEST_DIR) -vv --tb=long --capture=no

test-coverage:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

# Code quality targets
lint:
	ruff check $(SRC_DIR) $(TEST_DIR)

format:
	ruff check $(SRC_DIR) $(TEST_DIR) --fix
	ruff format $(SRC_DIR) $(TEST_DIR)

type-check:
	mypy $(SRC_DIR) --strict

check-all: lint type-check test
	@echo "✅ All checks passed!"

# Build targets
build: clean
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .pytest_cache/

# Release targets
release: check-all build
	@echo "Release $(VERSION) ready in dist/"

publish: build
	$(PYTHON) -m twine upload dist/*

# Development targets
dev-setup: install-dev
	@echo "✅ Development environment setup complete"

# Default target
.DEFAULT_GOAL := help
