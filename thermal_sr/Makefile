# ISRO Thermal SR Lab - Makefile

.PHONY: help install test train eval export clean data

help:
	@echo "ISRO Thermal SR Lab - Available Commands:"
	@echo ""
	@echo "  install     Install package and dependencies"
	@echo "  test        Run unit tests"
	@echo "  data        Generate synthetic test data"
	@echo "  train-cnn   Train Alignment-Fusion CNN model"
	@echo "  train-dis   Train Guidance-Disentanglement model"
	@echo "  train-swin  Train Swin Transformer model"
	@echo "  eval        Evaluate trained model"
	@echo "  export      Export model to ONNX"
	@echo "  clean       Clean generated files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

data:
	python sample_data/generate_synthetic.py --output-dir sample_data/synthetic --num-scenes 20

train-cnn:
	python scripts/train.py --config configs/train_cnn.yaml

train-dis:
	python scripts/train.py --config configs/train_disentangle.yaml

train-swin:
	python scripts/train.py --config configs/train_swin.yaml

eval:
	@echo "Usage: make eval MODEL=runs/best.pt DATA=sample_data/synthetic"
	@if [ -z "$(MODEL)" ] || [ -z "$(DATA)" ]; then \
		echo "Error: Please specify MODEL and DATA paths"; \
		exit 1; \
	fi
	python scripts/eval.py --checkpoint $(MODEL) --data $(DATA) --output results/

export:
	@echo "Usage: make export MODEL=runs/best.pt OUTPUT=models/model.onnx"
	@if [ -z "$(MODEL)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: Please specify MODEL and OUTPUT paths"; \
		exit 1; \
	fi
	python scripts/export_onnx.py --checkpoint $(MODEL) --output $(OUTPUT) --create-wrapper

infer:
	@echo "Usage: make infer MODEL=runs/best.pt OPTICAL=data/opt.tif THERMAL=data/th.tif OUTPUT=results/sr.tif"
	@if [ -z "$(MODEL)" ] || [ -z "$(OPTICAL)" ] || [ -z "$(THERMAL)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: Please specify MODEL, OPTICAL, THERMAL, and OUTPUT paths"; \
		exit 1; \
	fi
	python scripts/tile_infer.py --checkpoint $(MODEL) --optical $(OPTICAL) --thermal $(THERMAL) --output $(OUTPUT)

clean:
	rm -rf runs/
	rm -rf results/
	rm -rf sample_data/synthetic/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Quick start workflow
quickstart: install data train-cnn
	@echo "Quick start completed! Model trained on synthetic data."
	@echo "Run 'make eval MODEL=runs/cnn_thermal_sr/checkpoints/best_model.pt DATA=sample_data/synthetic' to evaluate."