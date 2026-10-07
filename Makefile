PYTHON ?= python

.PHONY: install test demo evaluate
install:
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m unittest discover -s test -v
demo:
	$(PYTHON) -m model_router demo --output results/demo.json
evaluate:
	$(PYTHON) -m model_router evaluate --output results/evaluation.json
