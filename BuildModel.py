import os
import boto3
from dotenv import load_dotenv


class BuildModel:
    def __init__(self):
        load_dotenv()  # Load environment variables
        # Set up instance variables for AWS credentials and region
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_default_region = os.getenv("AWS_DEFAULT_REGION")
        self.sagemaker_client = boto3.client(
            'sagemaker',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_default_region
        )
        # Debugging output to check if variables are loaded
        print("AWS Access Key:", os.getenv("AWS_ACCESS_KEY_ID"))
        print("AWS Secret Access Key:", os.getenv("AWS_SECRET_ACCESS_KEY"))
        print("Region:", os.getenv("AWS_DEFAULT_REGION"))

    def create_sagemaker_model(self, model_name, image_uri, model_data_url, role_arn):
        create_model_response = self.sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url
            }
        )
        print(f"Model created: {create_model_response['ModelArn']}")

    def deploy_model_to_endpoint(self, model_name, endpoint_config_name, endpoint_name):
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.m4.xlarge',
                    'InitialInstanceCount': 1
                }
            ]
        )
        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint {endpoint_name} is being created.")
