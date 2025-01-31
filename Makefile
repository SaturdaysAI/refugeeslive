#ENV_PATH = $(shell pyenv root)/versions/.venv
ENV_PATH = .venv

.PHONY: debug tests validate_code

# Run any pipeline located in src/pipelines
# Example: make evaluate_chatbot
.DEFAULT:
	$(ENV_PATH)/bin/python -m src.__main__ -n "$(@)"

# Debug any pipeline located in src/pipelines
# Example: make debug NAME=setup_system
debug:
	$(ENV_PATH)/bin/python -m pdb -m src.__main__ -n $(NAME)

# Run all project tests
tests:
	$(ENV_PATH)/bin/python -m unittest

# Check code syntax
validate_code:
	flake8 --exclude=./$(ENV_PATH),./build

# Build Docker image to deploy application locally or in AWS
build_image:
	docker build --platform linux/amd64 -t app .

# Deploy application from Docker image locally
local_deployment:
	docker run --rm -p 8000:8000 --entrypoint uvicorn --env-file .env --gpus all app src.app.app_api_handler:app --host 0.0.0.0 --port 8000

# Build AWS infrastructure and deploy application
infra_deployment:
	export $(cat .env | xargs)
	cd rag-cdk-infra; cdk deploy