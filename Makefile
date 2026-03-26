BOARD ?=
BOARD_MODELS_DIR ?= 

train:
	docker compose run --rm --no-deps trainer python scripts/train.py

finetune:
	docker compose run --rm --no-deps trainer python scripts/finetune.py

quantize:
	docker compose run --rm --no-deps vitis python scripts/quantize.py

compile:
	docker compose run --rm --no-deps vitis python scripts/compile.py

deploy:
	scp artifacts/drink_classifier.xmodel xilinx@ultra96:~/models/drink_classifier.xmodel

build:
	docker compose build

clean:
	docker compose down --remove-orphans --volumes

pipeline: build train finetune quantize compile deploy clean
