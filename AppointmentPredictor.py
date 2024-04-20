import boto3
import json


class AppointmentPredictor:
    def __init__(self):
        self.client = boto3.client('sagemaker-runtime',
                                   region_name='us-east-2')  # Replace 'us-east-1' with your desired AWS region
        self.endpoint_name = "PredictaCare"

    def predict_duration(self, patient_info):
        # Format the input data as JSON
        payload = json.dumps({
            'instances': [
                {
                    'name': patient_info['name'],
                    'age': patient_info['age'],
                    'gender': patient_info['gender'],
                    'description': patient_info['description'],
                    'is_new_patient': patient_info['is_new_patient']
                }
            ]
        })

        # Call the SageMaker endpoint
        response = self.client.invoke_endpoint(
            EndpointName="PredictaCare",
            Body=payload,
            ContentType='application/json'
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())

        # Assuming the result contains a field 'predicted_duration' with the prediction
        predicted_duration = result['predicted_duration']

        # Log the prediction for debugging
        print(
            f"Making prediction for {patient_info['name']}, age {patient_info['age']}, gender {patient_info['gender']}, new patient: {patient_info['is_new_patient']}")
        print(f"Predicted Duration: {predicted_duration}")

        return predicted_duration
