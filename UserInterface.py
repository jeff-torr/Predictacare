from AppointmentPredictor import AppointmentPredictor


class UserInterface:
    def __init__(self, name, age, gender, description, familiarity):
        self.name = name
        self.age = age
        self.gender = gender
        self.description = description
        self.familiarity = familiarity  # weather or not has been to respective


predictor = AppointmentPredictor('PredictaCare-Endpoint')
patient_info = UserInterface("John Doe", 29, "Male", "Regular checkup", True)
duration_prediction = predictor.predict_duration(patient_info)
print(f"Predicted Duration: {duration_prediction}")
