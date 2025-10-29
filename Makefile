.PHONY: install install-dev test test-cov lint format type-check clean build docs help

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  pre-commit   - Run pre-commit hooks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt
	pip install isort flake8

# Testing
test:
	pytest tests/

test-cov:
	pytest tests/ --cov=leibnetz --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast:
	pytest tests/ -x -v

# Code quality
lint:
	flake8 src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

# Pre-commit
pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean
	python -m build

# Documentation
docs:
	cd docs && make html

# Development workflow
dev-setup: install-dev pre-commit-install
	@echo "Development environment setup complete!"

# Run all checks (useful for CI or pre-commit)
check-all: lint type-check test-cov
	@echo "All checks passed!"

# Version info
version:
	python -c "import leibnetz; print(leibnetz.__version__)" 2>/dev/null || echo "Package not installed"
