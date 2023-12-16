import json
import boto3

s3_client = boto3.client('s3')
bucket_name = 'docs-upload-bucket'
content_type = 'application/pdf' # This could also come from the API Gateway HTTP headers.
expiration = '3600'

def lambda_handler(event, context):
    object_name = event['pathParameters']['object'] # Use the {object} path parameter
    response = s3_client.generate_presigned_url('put_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_name,
                                                        'ContentType': content_type
                                                },
                                                ExpiresIn=expiration,
                                                )

       
    return {
        'statusCode': 200,
        'headers': {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",  # Specify the allowed headers
            "Access-Control-Allow-Methods": "*",  # Specify the allowed HTTP methods
        },
        'body': json.dumps(response) # The response contains the presigned URL
    }