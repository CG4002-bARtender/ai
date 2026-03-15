BOARD ?= xilinx@ultra96
BOARD_MODELS_DIR ?= ~/models

pipeline:
	docker compose up --remove-orphans

train:
	docker compose run --rm --no-deps trainer python scripts/train.py

finetune:
	docker compose run --rm --no-deps trainer python scripts/finetune.py

quantize:
	docker compose run --rm --no-deps vitis python scripts/quantize.py

compile:
	docker compose run --rm --no-deps vitis python scripts/compile.py

deploy:
	scp artifacts/drink_classifier.xmodel $(BOARD):$(BOARD_MODELS_DIR)/drink_classifier.xmodel

build:
	docker compose build

clean:
	docker compose down --remove-orphans --volumes
