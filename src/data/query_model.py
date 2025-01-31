import os
import time
import uuid
import boto3
from pydantic import BaseModel, Field
from typing import List, Optional
from botocore.exceptions import ClientError

TABLE_NAME = os.environ.get("TABLE_NAME", "testTable")


class QueryModel(BaseModel):
    """
    Represents a query object stored in a DynamoDB table.

    Attributes:
        query_id (str): A unique identifier for the query, automatically generated 
            using a UUID when not explicitly provided.
        create_time (int): The Unix timestamp (in seconds) representing when the query 
            was created, automatically generated if not provided.
        query_text (str): The text of the query submitted by the user.
        answer_text (str, optional): The answer generated in response to the query. 
            Defaults to None if no answer is yet provided.
        is_complete (bool): Indicates whether the query has been processed and answered.
            Defaults to False.

    Methods:
        get_table() -> boto3.resource:
            Retrieves the DynamoDB table resource for storing and retrieving query objects.
        
        put_item() -> None:
            Inserts or updates the query object in the DynamoDB table.
        
        as_ddb_item() -> dict:
            Converts the query object's attributes into a dictionary suitable for 
            DynamoDB operations, excluding any attributes with None values.
        
        get_item(query_id: str) -> "QueryModel":
            Retrieves a query object by its ID from the DynamoDB table.
    """
    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    create_time: int = Field(default_factory=lambda: int(time.time()))
    query_text: str
    answer_text: Optional[str] = None
    is_complete: bool = False

    @classmethod
    def get_table(cls: "QueryModel") -> boto3.resource:
        """
        Retrieves the DynamoDB table resource specified by TABLE_NAME.

        Returns:
            boto3.resource: A reference to the DynamoDB Table.
        """
        dynamodb = boto3.resource("dynamodb")
        return dynamodb.Table(TABLE_NAME)

    def put_item(self) -> None:
        """
        Puts (inserts or replaces) the current QueryModel instance into the DynamoDB table.

        Raises:
            ClientError: If the call to DynamoDB fails.
        """
        item = self.as_ddb_item()
        try:
            response = QueryModel.get_table().put_item(Item=item)
            print(response)
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            raise e

    def as_ddb_item(self) -> dict:
        """
        Converts the QueryModel object into a dictionary that DynamoDB can store, 
        filtering out None values.

        Returns:
            dict: A dictionary representing the QueryModel suitable for DynamoDB.
        """
        item = {k: v for k, v in self.model_dump().items() if v is not None}
        return item

    @classmethod
    def get_item(cls: "QueryModel", query_id: str) -> "QueryModel":
        """
        Fetches a QueryModel object from DynamoDB by its query_id.

        Args:
            query_id (str): The unique identifier for the query.

        Returns:
            QueryModel or None: The corresponding QueryModel if found, otherwise None.
        """
        try:
            response = cls.get_table().get_item(Key={"query_id": query_id})
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            return None

        if "Item" in response:
            item = response["Item"]
            return cls(**item)
        else:
            return None
