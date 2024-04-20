import boto3
import sagemaker
from sagemaker.model import Model


class DeployModel:
    def __init__(self, model_name, image_uri, role, model_data):
        # Create a Boto3 session with a specific region
        boto_session = boto3.Session(region_name='us-east-2')  # Replace 'us-east-2' with your desired AWS region

        # Initialize SageMaker session using the Boto3 session
        self.sagemaker_session = sagemaker.Session(boto_session=boto_session)

        # Create a Model object
        self.sagemaker_model = Model(
            image_uri=image_uri,
            model_data=model_data,
            role=role,
            sagemaker_session=self.sagemaker_session,
            name=model_name
        )

        # Deploy the model to an endpoint automatically when an instance is created
        self.predictor = self.sagemaker_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large'
        )
        print(f"Deployment complete. Endpoint {self.predictor.endpoint_name} is now active.")
