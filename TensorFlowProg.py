import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')


class DataHandler:
    def __init__(self, filename):
        self.filename = "Estimate Healthcare Appointment Length Given X - Sheet1.csv"
        self.ensure_nltk_packages()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def ensure_nltk_packages(self):
        nltk_packages = ['punkt', 'wordnet', 'stopwords']
        for package in nltk_packages:
            try:
                nltk.data.find(f'tokenizers/punkt/{package}.pickle')
            except LookupError:
                nltk.download(package)

    def load_data(self):
        data = pd.read_csv(self.filename)
        sentences = data.iloc[:, 0].values
        times = data.iloc[:, 1].values
        return sentences, times

    def preprocess_data(self, sentences, tokenizer):
        processed_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            lemmatized = [self.lemmatizer.lemmatize(word.lower()) for word in words if
                          word.lower() not in self.stop_words]
            processed_sentences.append(' '.join(lemmatized))
        tokenizer.fit_on_texts(processed_sentences)
        sequences = tokenizer.texts_to_sequences(processed_sentences)
        padded_sequences = pad_sequences(sequences, maxlen=100)
        return padded_sequences


class TokenizerManager:
    @staticmethod
    def save(tokenizer, path='tokenizer.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path='tokenizer.pickle'):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


class ModelTrainer:
    def __init__(self, vocab_size, embedding_dim=50, max_length=100):
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1, activation='linear')
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    def train(self, X_train, y_train):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}, Test MAE: {mae}')

    def save_model(self, path='appointment_duration_model.h5'):
        self.model.save(path)


class Predictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = TokenizerManager.load(tokenizer_path)

    def predict_duration(self, description):
        sequence = self.tokenizer.texts_to_sequences([description])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        predicted_duration = self.model.predict(padded_sequence)
        return predicted_duration[0][0]


def main():
    # Setup
    data_file = 'Estimate Healthcare Appointment Length Given X - Sheet1.csv'
    model_file = 'appointment_duration_model.h5'
    tokenizer_file = 'tokenizer.pickle'

    # Load and preprocess data
    data_handler = DataHandler(data_file)
    sentences, times = data_handler.load_data()
    tokenizer = Tokenizer()
    padded_sequences = data_handler.preprocess_data(sentences, tokenizer)
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, times, test_size=0.2, random_state=0)

    # Train the model
    vocab_size = len(tokenizer.word_index) + 1
    trainer = ModelTrainer(vocab_size)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.save_model(model_file)

    # Save tokenizer
    TokenizerManager.save(tokenizer, tokenizer_file)

    # Predicting with the model
    predictor = Predictor(model_file, tokenizer_file)
    test_description = "Neonatal examination for infant screening"
    predicted_duration = predictor.predict_duration(test_description)
    print(f"Predicted Duration: {predicted_duration:.2f} minutes")


if __name__ == "__main__":
    main()