import pandas as pd
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class DataHandler:
    def __init__(self, filename):
        self.filename = filename
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
        gender = data.iloc[:, 0].values
        age = data.iloc[:, 1].values.astype(int)
        sentences = data.iloc[:, 2].values
        durations = data.iloc[:, 3].values.astype(int)
        familiarity = data.iloc[:, 4].values
        return gender, age, sentences, durations, familiarity

    def preprocess_data(self, sentences):
        processed_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            lemmatized = [self.lemmatizer.lemmatize(word.lower())
                          for word in words if word.lower() not in self.stop_words]
            processed_sentences.append(' '.join(lemmatized))
        return processed_sentences


class LinearRegressionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.model = LinearRegression()

    def train(self, X_train_text, X_train_features, y_train):
        X_train_counts = self.vectorizer.fit_transform(X_train_text)
        X_train_combined = hstack([X_train_counts, X_train_features])
        self.model.fit(X_train_combined, y_train)

    def predict(self, descriptions, additional_features):
        counts = self.vectorizer.transform(descriptions)
        X_input = hstack([counts, additional_features])
        predictions = self.model.predict(X_input)
        return predictions

    def evaluate(self, X_test_text, X_test_features, y_test):
        X_test_counts = self.vectorizer.transform(X_test_text)
        X_test_combined = hstack([X_test_counts, X_test_features])
        predictions = self.model.predict(X_test_combined)
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Mean Squared Error: {mse}")

    def save(self, path='linear_regression_model.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The CSV file containing the training data")
    parser.add_argument("user_input", help="The description for which to predict the appointment length")
    parser.add_argument("age_input", help="The user's age")
    parser.add_argument("gender_input", help="The user's gender")
    parser.add_argument("familiarity_input", help="Has the user been here before?")
    args = parser.parse_args()

    data_file = args.filename
    user_input = args.user_input
    age_input = int(args.age_input)  # Convert to int
    gender_input = 0 if args.gender_input.lower() == 'female' else 1  # Convert to numeric
    familiarity_input = 1 if args.familiarity_input.lower() == 'yes' else 0  # Convert to numeric

    # Load and preprocess data
    data_handler = DataHandler(data_file)
    gender, age, sentences, durations, familiarity = data_handler.load_data()
    processed_sentences = data_handler.preprocess_data(sentences)

    # Encode gender and familiarity to numeric
    gender_encoded = np.where(gender == "Female", 0, 1)
    familiarity_encoded = np.where(familiarity == "Yes", 1, 0)

    # Combine features into a single matrix
    additional_features = pd.DataFrame({
        'gender': gender_encoded,
        'age': age,
        'familiarity': familiarity_encoded
    }).astype(float)  # Ensure the features are of type float

    # Split the dataset
    X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
        processed_sentences, additional_features.values, durations, test_size=0.2, random_state=50)

    # Train and evaluate the model
    model = LinearRegressionModel()
    model.train(X_train_text, X_train_features, y_train)
    # model.evaluate(X_test_text, X_test_features, y_test)
    model.save()

    # Process the user input for prediction
    processed_input = data_handler.preprocess_data([user_input])
    additional_features_input = pd.DataFrame({
        'gender': [gender_input], 
        'age': [age_input],
        'familiarity': [familiarity_input]
    }).astype(float)  # Ensure the features are of type float

    predicted_duration = model.predict(processed_input, additional_features_input.values)
    print(predicted_duration[0])


if __name__ == "__main__":
    main()



#old model:

# import pandas as pd
# import pickle
# import argparse
# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# class DataHandler:
#     def __init__(self, filename):
#         self.filename = filename
#         self.ensure_nltk_packages()
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(stopwords.words('english'))

#     def ensure_nltk_packages(self):
#         nltk_packages = ['punkt', 'wordnet', 'stopwords']
#         for package in nltk_packages:
#             try:
#                 nltk.data.find(f'tokenizers/punkt/{package}.pickle')
#             except LookupError:
#                 nltk.download(package)

#     def load_data(self):
#         data = pd.read_csv(self.filename)
#         sentences = data.iloc[:, 0].values
#         durations = data.iloc[:, 1].values.astype(int)
#         return sentences, durations

#     def preprocess_data(self, sentences):
#         processed_sentences = []
#         for sentence in sentences:
#             words = nltk.word_tokenize(sentence)
#             lemmatized = [self.lemmatizer.lemmatize(word.lower())
#                           for word in words if word.lower() not in self.stop_words]
#             processed_sentences.append(' '.join(lemmatized))
#         return processed_sentences

# class NaiveBayesModel:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#         self.model = MultinomialNB(alpha=0.1)

#     def train(self, X_train, y_train):
#         X_train_counts = self.vectorizer.fit_transform(X_train)
#         self.model.fit(X_train_counts, y_train)

#     def predict(self, descriptions):
#         counts = self.vectorizer.transform(descriptions)
#         return self.model.predict(counts)

#     def evaluate(self, X_test, y_test):
#         X_test_counts = self.vectorizer.transform(X_test)
#         predictions = self.model.predict(X_test_counts)

#     def save(self, path='naive_bayes_model.pickle'):
#         with open(path, 'wb') as handle:
#             pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("filename", help="The CSV file containing the training data")
#     parser.add_argument("user_input", help="The description for which to predict the appointment length")
#     args = parser.parse_args()

#     data_file = args.filename
#     user_input = args.user_input

#     # Load and preprocess data
#     data_handler = DataHandler(data_file)
#     sentences, durations = data_handler.load_data()
#     processed_sentences = data_handler.preprocess_data(sentences)

#     # Split the dataset
#     X_train, X_test, y_train, y_test = train_test_split(processed_sentences, durations, test_size=0.2, random_state=50)

#     # Train Naive Bayes Model
#     nb_model = NaiveBayesModel()
#     nb_model.train(X_train, y_train)
#     nb_model.evaluate(X_test, y_test)
#     nb_model.save()

#     processed_input = data_handler.preprocess_data([user_input])
#     predicted_duration = nb_model.predict(processed_input)
#     print(predicted_duration[0])

# if __name__ == "__main__":
#     main()