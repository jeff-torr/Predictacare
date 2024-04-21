from AppointmentPredictor import AppointmentPredictor
from BuildModel import BuildModel
from DeployModel import DeployModel
from UserInterface import UserInterface

def main():
    # Assuming all required variables are defined within the classes or passed correctly
    builder = BuildModel()
    builder.create_sagemaker_model(
        model_name='PredictaCare-Model',
        image_uri='257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.3-1',
        model_data_url='s3://predictacosttraining/output/PredictaCare/output/model.tar.gz',
        role_arn='arn:aws:iam::767398032002:role/service-role/AmazonSageMaker-ExecutionRole-20240420T142608'
    )
    builder.deploy_model_to_endpoint(
        model_name='PredictaCare-Model',
        endpoint_config_name='PredictaCare-EndpointConfig',
        endpoint_name='PredictaCare-Endpoint'
    )

    # Assuming 'DeployModel' takes these arguments in its constructor
    deployer = DeployModel('PredictaCare-Model', 'image_uri_placeholder', 'role_placeholder', 'model_data_placeholder')

    # Helper function to simplify user input
    def get_input(prompt, cast_type=str, condition=lambda x: True, error_message="Invalid input, please try again."):
        while True:
            user_input = input(prompt)
            try:
                value = cast_type(user_input)
                if condition(value):
                    return value
                else:
                    print(error_message)
            except ValueError:
                print(error_message)

    name = get_input("Patient's name? ")
    age = get_input("Patient's age? ", int)
    gender = get_input("Patient's gender? ")
    description = get_input("Description of appointment: ")
    familiarity = get_input("Are they a new patient (Y/N)? ", str, lambda x: x.upper() in ['Y', 'N'])

    user = UserInterface(name, age, gender, description, familiarity == 'Y')

if __name__ == "__main__":
    main()