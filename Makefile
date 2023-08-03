TITLE:="Makefile for turbines predictive maintenace"

help:
	@echo [$(TITLE)]
	@echo "=> Commands:"
	@echo "make setup        - Create virtual environment and install dependencies"
	@echo "make requirements - Create requirements.txt"
	@echo "make clean        - Remove virtual environment and create requirements.txt"

setup:
	@echo "Creating virtual environment..."
	python3.8 -m virtualenv venv
	@echo "Installing dependencies..."
	venv/bin/pip install -r requirements.txt
	@echo "Starting VSCode..."
	code .

requirements:
	@echo "Creating requirements.txt..."
	venv/bin/pip freeze > requirements.txt

clean: requirements
	@echo "Removing virtual environment..."
	rm -rf ./src/__pycache__
	rm -rf venv
