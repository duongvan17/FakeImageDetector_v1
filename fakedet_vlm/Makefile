.PHONY: help install verify test data stage1 stage2 eval serve docker

help:
	@echo "  make install   - editable install with dev+train extras"
	@echo "  make verify    - sanity-check ViT loader on real checkpoint"
	@echo "  make test      - run pytest suite"
	@echo "  make data      - build LLaVA JSON splits from FakeClue"
	@echo "  make stage1    - run stage 1 projector alignment"
	@echo "  make stage2    - run stage 2 LoRA SFT"
	@echo "  make eval      - evaluate trained model on val set"
	@echo "  make serve     - launch FastAPI server on :8000"
	@echo "  make docker    - build Docker image"

install:
	pip install -e ".[dev,train]"

verify:
	python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth

test:
	pytest -q

data:
	python scripts/prepare_fakeclue.py --out data --train-ratio 0.9

stage1:
	python -m fakedet_vlm.train.stage1 --base configs/base.yaml --stage configs/stage1.yaml

stage2:
	python -m fakedet_vlm.train.stage2 --base configs/base.yaml --stage configs/stage2.yaml

eval:
	python scripts/eval.py \
	  --val-json data/val.json --images-dir data/images \
	  --adapter-dir runs/stage2_sft/final \
	  --projector runs/stage2_sft/final/projector.pt \
	  --out runs/stage2_sft/eval

serve:
	uvicorn fakedet_vlm.serve.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t fakedet-vlm:0.1 .
