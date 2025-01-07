import json
import boto3
import pandas as pd


def validate_data(event, context):
    s3 = boto3.client('s3')

    # Get file info from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download and validate file
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])

    # Validation logic
    required_columns = ['sequence_id', 'heavy_chain', 'light_chain', 'binding_affinity']
    validation_passed = all(col in df.columns for col in required_columns)

    if validation_passed:
        # Trigger AWS Batch job
        batch = boto3.client('batch')
        response = batch.submit_job(
            jobName='antibody-pipeline',
            jobQueue='antibody-pipeline-queue',
            jobDefinition='antibody-pipeline',
            parameters={
                'inputFile': f's3://{bucket}/{key}',
                'outputFile': f's3://antibody-pipeline-output/processed_{key}'
            }
        )
        return {
            'statusCode': 200,
            'body': json.dumps(f'Validation passed, triggered job {response["jobId"]}')
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Validation failed: missing required columns')
        }