.PHONY: help install format lint test clean

ENV = parafac_traffic

help:
	@echo "Commands"
	@echo "========"
	@echo "  make install   Create conda environment and install dependencies"
	@echo "  make format    Format code (black + isort)"
	@echo "  make lint      Run flake8 + pylint"
	@echo "  make test      Run tests"
	@echo "  make clean     Remove cache files"

install:
	conda create -n $(ENV) python=3.11 -y
	conda run -n $(ENV) pip install -e ".[dev]"
	@echo ""
	@echo "Done. Activate with: conda activate $(ENV)"

format:
	conda run -n $(ENV) isort --profile black --line-length 120 .
	conda run -n $(ENV) black --line-length 120 .

lint:
	conda run -n $(ENV) flake8 . --max-line-length=120 --exclude=.ipynb_checkpoints,data
	conda run -n $(ENV) pylint train_model.py --output-format=colorized

test:
	conda run -n $(ENV) pytest -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
