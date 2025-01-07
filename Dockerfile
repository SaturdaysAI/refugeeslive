FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY config/requirements_app.txt ${LAMBDA_TASK_ROOT}

# Required to make SQLlite3 work for Chroma.
RUN pip install pysqlite3-binary

# Install the specified packagess
RUN pip install -r requirements_app.txt

# For local testing
EXPOSE 8000

# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True

# Copy files in ./src
COPY src/app/app_api_handler.py ${LAMBDA_TASK_ROOT}/src/app/app_api_handler.py
COPY src/data/get_chroma_db.py ${LAMBDA_TASK_ROOT}/src/data/get_chroma_db.py
COPY src/data/prompts.py ${LAMBDA_TASK_ROOT}/src/data/prompts.py
COPY src/data/query_model.py ${LAMBDA_TASK_ROOT}/src/data/query_model.py
COPY src/models/chatbot.py ${LAMBDA_TASK_ROOT}/src/models/chatbot.py
COPY src/tools/startup.py ${LAMBDA_TASK_ROOT}/src/tools/startup.py
COPY src/tools/utils.py ${LAMBDA_TASK_ROOT}/src/tools/utils.py
COPY googletrans ${LAMBDA_TASK_ROOT}/googletrans
COPY data/interim/vectorstore ${LAMBDA_TASK_ROOT}/data/interim/vectorstore
COPY data/models ${LAMBDA_TASK_ROOT}/data/models
COPY config/main.yaml ${LAMBDA_TASK_ROOT}/config/main.yaml
COPY config/settings.yaml ${LAMBDA_TASK_ROOT}/config/settings.yaml

RUN mkdir ${LAMBDA_TASK_ROOT}/data/logs
