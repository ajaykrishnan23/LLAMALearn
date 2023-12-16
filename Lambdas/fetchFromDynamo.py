import json
import boto3
from boto3.dynamodb.conditions import Key, Attr


def lambda_handler(event, context):

    # Initialize DynamoDB client
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  # Replace 'your-region' with your AWS region
    # Specify the table name
    table_name = 'user-qna'  # Replace with your DynamoDB table name
    # Get the DynamoDB table
    table = dynamodb.Table(table_name)
    
    # Perform the query
    response = table.query(
        KeyConditionExpression=Key('uqid').eq(event['queryStringParameters']['userId']) 
    )
    
    items = response['Items']
    print(items)
    
    if items:
        # Process the data as needed (e.g., return it as JSON)
        return {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",  # Specify the allowed headers
                "Access-Control-Allow-Methods": "*",  # Specify the allowed HTTP methods
            },
            'body': json.dumps(items)
        }
    else:
        return {
            'statusCode': 404,
            'headers': {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",  # Specify the allowed headers
                "Access-Control-Allow-Methods": "*",  # Specify the allowed HTTP methods
            },
            'body': json.dumps({'error': 'Item not found'})
        }
