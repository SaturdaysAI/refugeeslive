import os
import boto3
import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.data.query_model import QueryModel
from src.models.chatbot import RAGRefugeesChatbot
from src.tools.startup import params as settings


WORKER_LAMBDA_NAME = os.environ.get("WORKER_LAMBDA_NAME", None)
AWS_DEPLOYMENT = bool(os.environ.get("AWS_DEPLOYMENT", False))

app = FastAPI()

if not AWS_DEPLOYMENT:
    chatbot = RAGRefugeesChatbot(settings["generate_answers"]["params"])


class SubmitQueryRequest(BaseModel):
    """
    A Pydantic model representing a user's query request payload.
    
    Attributes:
        query_text (str): The text of the query submitted by the user.
    """
    query_text: str


def invoke_rag(request: SubmitQueryRequest, chatbot: object) -> str:
    """
    Invokes the RAGRefugeesChatbot to obtain an answer to the user query.
    
    Args:
        request (SubmitQueryRequest): A request object containing the user's
        query text.
        
    Returns:
        str: The chatbot's response to the submitted query.
    """
    query_response = chatbot.answer_question(request.query_text)
    return query_response


def invoke_worker(query: QueryModel) -> None:
    """
    Invokes another Lambda function (the 'worker') asynchronously,
    passing in the serialized QueryModel object as payload.
    
    Args:
        query (QueryModel): An instance of the QueryModel containing query
        details.
    """
    # Initialize the Lambda client
    lambda_client = boto3.client("lambda")

    # Get the QueryModel as a dictionary
    payload = query.model_dump()

    # Invoke another Lambda function asynchronously
    response = lambda_client.invoke(
        FunctionName=WORKER_LAMBDA_NAME,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )

    print(f"✅ Worker Lambda invoked: {response}")


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryModel:
    """
    An endpoint to submit a user query. Depending on the deployment type (local 
    or in AWS), this may invoke a worker Lambda asynchronously or process the
    request synchronously to use the RAG chatbot to generate a response.
    
    Args:
        request (SubmitQueryRequest): The query payload from the user.
        
    Returns:
        QueryModel: The persisted query object, which includes 
        the answer if it was processed synchronously.
    """
    new_query = QueryModel(query_text=request.query_text)

    if AWS_DEPLOYMENT:
        # Make an async call to the worker.
        new_query.put_item()
        invoke_worker(new_query)
    else:
        # Make a synchronous call to the worker.
        response_text = invoke_rag(request, chatbot)
        new_query.answer_text = response_text
        new_query.is_complete = True

    return new_query


def lambda_submit_query_handler(event, context) -> dict:
    """
    AWS Lambda handler to receive and process a 'submit query' event
    from API Gateway. Creates a new query and returns the result.
    
    Args:
        event (dict): The event payload from API Gateway, containing 
            - "body" (str): JSON string with the user's query.
        context: AWS Lambda context object (not used here).
        
    Returns:
        dict: A response containing the status code and 
        the newly created query details as JSON.
    """
    event_body = json.loads(event["body"])
    submit_query_request = SubmitQueryRequest(query_text=event_body["query"])
    new_query = submit_query_endpoint(submit_query_request)
    print("new query created")
    return {
        "statusCode": 200,
        "body": json.dumps(new_query.model_dump())
    }


def lambda_worker_handler(event, context) -> None:
    """
    AWS Lambda handler for the worker Lambda. 
    Processes the received query and updates it with an answer
    from the RAG chatbot, then persists the updated query.
    
    Args:
        event (dict): A dictionary representation of the QueryModel.
        context: AWS Lambda context object (not used here).
    """
    query_item = QueryModel(**event)
    submit_query_request = SubmitQueryRequest(query_text=query_item.query_text)
    chatbot_instance = RAGRefugeesChatbot(settings["generate_answers"]["params"])
    response_text = invoke_rag(submit_query_request, chatbot_instance)
    query_item.answer_text = response_text
    query_item.is_complete = True
    query_item.put_item()
    print(f"✅ Item is updated: {query_item}")


def lambda_get_query_handler(event, context) -> dict:
    """
    AWS Lambda handler for retrieving a query record.
    
    Args:
        event (dict): The event payload from API Gateway, containing 
            - "queryStringParameters" (dict): Dictionary with "query_id" key.
        context: AWS Lambda context object (not used here).
        
    Returns:
        dict: A response containing the status code and the serialized query
        details as JSON.
    """
    event_body = event["queryStringParameters"]
    query = QueryModel.get_item(event_body["query_id"])
    return {
        "statusCode": 200,
        "body": json.dumps(query.model_dump())
    }


if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)
