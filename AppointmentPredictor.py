import boto3
import json


class AppointmentPredictor:
    def __init__(self, endpoint_name):
        self.client = boto3.client('sagemaker-runtime')
        self.endpoint_name = endpoint_name

    def create_sagemaker_model(model_name, image_uri, model_data_url, role_arn):
        sagemaker_client = boto3.client('sagemaker')
        create_model_response = sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url
            }
        )
        print(f"Model created: {create_model_response['ModelArn']}")

    create_sagemaker_model(
        model_name='PredictaCare-Model',
        image_uri='257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.3-1',
        model_data_url='s3://predictacosttraining/path/to/model.tar.gz',
        role_arn='arn:aws:iam::767398032002:role/service-role/AmazonSageMaker-ExecutionRole-20240420T142608'
    )

    def deploy_model_to_endpoint(model_name, endpoint_config_name, endpoint_name):
        sagemaker_client = boto3.client('sagemaker')
        sagemaker_client.create_endpoint_config(
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
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint {endpoint_name} is being created.")

    deploy_model_to_endpoint(
        model_name='PredictaCare-Model',
        endpoint_config_name='PredictaCare-EndpointConfig',
        endpoint_name='PredictaCare-Endpoint'
    )

    def predict_duration(self, patient):
        # Serialize the patient data into JSON or your required format
        patient_data = {
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "description": patient.description,
            "is_new_patient": patient.is_new_patient
        }
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(patient_data)
        )

        result = response['Body'].read()
        return json.loads(result)
