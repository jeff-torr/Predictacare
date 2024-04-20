from AppointmentPredictor import AppointmentPredictor


class UserInterface:
    def __init__(self, name, age, gender, description, is_new_patient):
        self.name = name
        self.age = age
        self.gender = gender
        self.description = description
        self.is_new_patient = is_new_patient  # Whether the patient has been to the respective doctor before

    def get_patient_info(self):
        # Return patient information in a dictionary format expected by AppointmentPredictor
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "description": self.description,
            "is_new_patient": self.is_new_patient
        }


# Initialize the predictor
predictor = AppointmentPredictor()

# Create a UserInterface object for the patient
patient_ui = UserInterface("John Doe", 29, "Male", "Regular checkup", True)

# Retrieve patient info and predict duration
patient_info = patient_ui.get_patient_info()
# duration_prediction = predictor.predict_duration(patient_info)

# # Print the predicted duration
# print(f"Predicted Duration: {duration_prediction}")
