import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class TokenizerManager:
    @staticmethod
    def load(path='tokenizer.pickle'):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class Predictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = TokenizerManager.load(tokenizer_path)

    def predict_duration(self, description):
        sequence = self.tokenizer.texts_to_sequences([description])
        print(f"Sequence: {sequence}")  # Debugging: Check the sequence
        padded_sequence = pad_sequences(sequence, maxlen=16)
        print(f"Padded Sequence: {padded_sequence}")  # Debugging: Check the padded sequence
        predicted_duration = self.model.predict(padded_sequence)
        return predicted_duration[0][0]


def main():
    # Setup
    model_file = 'appointment_duration_model.h5'
    tokenizer_file = 'tokenizer.pickle'

    # Load model and tokenizer
    predictor = Predictor(model_file, tokenizer_file)

    # Continuous prediction loop
    while True:
        user_input = input("Enter a description of the patient's ailment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predicted_duration = predictor.predict_duration(user_input)
        print(f"Predicted Duration: {predicted_duration:.2f} minutes")


if __name__ == "__main__":
    main()
