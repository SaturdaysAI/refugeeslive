import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';

import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import {
  DockerImageFunction,
  DockerImageCode,
  Architecture,
} from "aws-cdk-lib/aws-lambda";
import * as apigateway from "aws-cdk-lib/aws-apigateway"
import { ManagedPolicy } from "aws-cdk-lib/aws-iam";
import { Platform } from 'aws-cdk-lib/aws-ecr-assets';

export class RagCdkInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create a DynamoDB table to store the query data and results.
    const ragQueryTable = new Table(this, "RagQueryTable", {
      partitionKey: { name: "query_id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
    });
    console.log(Date(), "Created a DynamoDB table")

    // Lambda worker function of the submit_query endpoint
    const workerImageCode = DockerImageCode.fromImageAsset("..", {
      cmd: ["src.app.app_api_handler.lambda_worker_handler"],
      platform: Platform.LINUX_AMD64,
      });
    const workerFunction = new DockerImageFunction(this, "workerFunc", {
      code: workerImageCode,
      memorySize: 1000,
      timeout: cdk.Duration.seconds(60),
      architecture: Architecture.X86_64,
      environment: {
        TABLE_NAME: ragQueryTable.tableName,
        AWS_DEPLOYMENT: "True",
      },
    });
    console.log(Date(), "Created lambda worker function")

    // Lambda function to handle the "POST/submit_query" endpoint
    const submitQueryImageCode = DockerImageCode.fromImageAsset("..", {
      cmd: ["src.app.app_api_handler.lambda_submit_query_handler"],
      platform: Platform.LINUX_AMD64,
    });
    const submitQueryFunction = new DockerImageFunction(this, "submitQueryFunc", {
      code: submitQueryImageCode,
      memorySize: 200,
      timeout: cdk.Duration.seconds(10),
      architecture: Architecture.X86_64,
      environment: {
        TABLE_NAME: ragQueryTable.tableName,
        WORKER_LAMBDA_NAME: workerFunction.functionName,
        AWS_DEPLOYMENT: "True"
      },
    });
    console.log(Date(), "Created lambda submit query function")

    // Lambda function to handle the "GET/get_query" endpoint
    const getQueryImageCode = DockerImageCode.fromImageAsset("..", {
      cmd: ["src.app.app_api_handler.lambda_get_query_handler"],
      platform: Platform.LINUX_AMD64,
    });
    const getQueryFunction = new DockerImageFunction(this, "getQueryFunc", {
      code: getQueryImageCode,
      memorySize: 200,
      timeout: cdk.Duration.seconds(10),
      architecture: Architecture.X86_64,
      environment: {
        TABLE_NAME: ragQueryTable.tableName,
        AWS_DEPLOYMENT: "True",
      },
    });
    console.log(Date(), "Created lambda get query function")

    // API Gateway REST API
    const api = new apigateway.RestApi(this, "RestApi", {
      restApiName: "RefugeesChatbot",
      description: "API Gateway for the Refugees Chatbot with API key access control.",
    //  policy: apiResourcePolicy
    });
    console.log(Date(), "Created api gateway rest api")

    // Create an API key needed to grant access to the API resources
    const apiKey = api.addApiKey('ApiKey', {
      apiKeyName: 'HuggingFaceApiKey',
      description: "API key to access the API Gateway methods from HuggingFace"
    });
    console.log(Date(), "Created api key")

    const usagePlan = api.addUsagePlan('UsagePlan', {
      name: 'BasicUsagePlan',
      throttle: {
        rateLimit: 10,
        burstLimit: 2,
      },
      quota: {
        limit: 1000,
        period: apigateway.Period.DAY,
      },
    });

    usagePlan.addApiKey(apiKey);
    usagePlan.addApiStage({
      stage: api.deploymentStage,
    });
    console.log(Date(), "Created and assigned usage plan")

    // Add "submit_query" resource
    const submit_query = api.root.addResource('submit_query');
    // Assign POST method to the "submitQueryFunction" Lambda
    submit_query.addMethod("POST",
      new apigateway.LambdaIntegration(submitQueryFunction),{
        apiKeyRequired: true
      });

    // Add "get_query" resource
    const get_query = api.root.addResource('get_query');
    // Assign GET method to the "getQueryFunction" Lambda
    get_query.addMethod("GET",
      new apigateway.LambdaIntegration(getQueryFunction), {
        apiKeyRequired: true
      });

    api.addGatewayResponse('IPRestrictPolicy', {
      type: apigateway.ResponseType.UNAUTHORIZED,
      responseHeaders: {
        "Access-Control-Allow-Origin": "'*'",
        "Access-Control-Allow-Credentials": "'true'"
      }
    });
    
    console.log(Date(), "Added submit_query and get_query resources to api gateway")

    // Grant permissions for all resource to work together
    ragQueryTable.grantReadWriteData(workerFunction);
    ragQueryTable.grantReadWriteData(submitQueryFunction);
    ragQueryTable.grantReadWriteData(getQueryFunction);
    workerFunction.grantInvoke(submitQueryFunction);
    workerFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );

    console.log(Date(), "Grant permissions for all resources")

    // Output the API URL to the console after deployment
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.url ?? 'Something went wrong with the deployment',
      description: 'The URL of the API Gateway',
      exportName: 'RestApiUrl',
    });
  }
}