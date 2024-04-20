import boto3
import json


class AppointmentPredictor:
    def __init__(self, endpoint_name):
        self.client = boto3.client('sagemaker-runtime')
        self.endpoint_name = endpoint_name

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
