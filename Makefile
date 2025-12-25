.PHONY: install test lint format clean run-demo train-adapter train-router deploy docker-build

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Demo & Serving
run-demo:
	python scripts/run_gradio_server.py

run-api:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

# Training
train-adapter:
	python -m src.adapters.lora_trainer \
		--adapter_config configs/adapters/$(ADAPTER)_adapter.yaml \
		--output_dir models/$(ADAPTER)

train-router:
	python experiments/train_router.py \
		--config configs/router_config.yaml \
		--output_dir models/router

# Data
prepare-data:
	python scripts/prepare_data.py --adapter $(ADAPTER) --samples $(SAMPLES)

generate-synthetic:
	python -m src.data.synthetic_generator \
		--output_dir data/synthetic \
		--samples 1000

# Docker
docker-build:
	docker build -t adaptive-lora:latest .

docker-run:
	docker run -p 8000:8000 --gpus all adaptive-lora:latest

docker-compose-up:
	docker-compose -f docker-compose.prod.yml up -d

docker-compose-down:
	docker-compose -f docker-compose.prod.yml down

# Kubernetes
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf dist build *.egg-info

# Help
help:
	@echo "Available targets:"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install dev dependencies"
	@echo "  test           - Run all tests"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  run-demo       - Start Gradio demo"
	@echo "  run-api        - Start FastAPI server"
	@echo "  train-adapter  - Train LoRA adapter (ADAPTER=reasoning)"
	@echo "  train-router   - Train router network"
	@echo "  docker-build   - Build Docker image"
	@echo "  k8s-deploy     - Deploy to Kubernetes"
	@echo "  clean          - Clean build artifacts"
