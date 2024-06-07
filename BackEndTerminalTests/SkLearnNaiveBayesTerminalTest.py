# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LinearRegression
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
#         gender = data.iloc[:, 0].values
#         age = data.iloc[:, 1].values.astype(int)
#         sentences = sentences = data.iloc[:, 2].values
#         durations = data.iloc[:, 3].values.astype(int)

#         # sentences = data.iloc[:, 0].values
#         # # assign appointment length time to int
#         # durations = data.iloc[:, 1].values.astype(int)
#         return gender, age, sentences, durations

#     def preprocess_data(self, sentences):
#         processed_sentences = []
#         for sentence in sentences:
#             words = nltk.word_tokenize(sentence)
#             # break word down to its root word/meaning
#             lemmatized = [self.lemmatizer.lemmatize(word.lower())
#                           for word in words if word.lower() not in self.stop_words]
#             processed_sentences.append(' '.join(lemmatized))
#         return processed_sentences


# class LinearRegressionModel:
#     def __init__(self):
#         #instantiate model Class objs
#         self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#         # self.model = LinearRegression(alpha=0.1)
#         self.model = LinearRegression()

#     def train(self, X_train, y_train):
#         # X is matrix of samples to features
#         # y is array of samples
#         # statistic model to find word relevance
#         X_train_counts = self.vectorizer.fit_transform(X_train)
#         # fit model with Tfidvectors
#         self.model.fit(X_train_counts, y_train)

#     def predict(self, descriptions, age, gender):
#         counts = self.vectorizer.transform(descriptions, age, gender)
#         return self.model.predict(counts)

#     def evaluate(self, X_test, y_test):
#         # returns matrix of n samples, n features
#         X_test_counts = self.vectorizer.transform(X_test)
#         predictions = self.model.predict(X_test_counts)
#         # accuracy = accuracy_score(y_test, predictions)

#     def save(self, path='naive_bayes_model.pickle'):
#         with open(path, 'wb') as handle:
#             pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)


# def main():
#     data_file = 'EstimateHealthcareAppointmentLengthGivenX-Sheet1.csv'

#     # load and preprocess data
#     data_handler = DataHandler(data_file)
#     gender, age, sentences, durations = data_handler.load_data()
#     processed_sentences = data_handler.preprocess_data(sentences)

#     #rearange so splitting rest of variables
#     #have a column for gender age
#     #gender convert into 0s and 1s
#     #turn sentecnes into matrix where each row is training ex and column is count of number of words in doc
#     #append to matrix columns for gender and age
    
#     for i in range(len(gender)):
#         if gender[i] == "Female":
#             gender[i] = 0
#         elif gender[i] == "Male":
#             gender[i] = 1
    


#     # split the dataset
#     X_train, X_test, y_train, y_test = train_test_split(gender, age, processed_sentences, durations, test_size=0.2, random_state=50)

#     # train Naive Bayes Model
#     lr_model = LinearRegressionModel()
#     lr_model.train(X_train, y_train)
#     lr_model.evaluate(X_test, y_test)
#     lr_model.save()

#     while True:
#         user_input = input("Enter an appointment description (or type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             break
#         gender_input = input("Enter gender (0 for Female, 1 for Male): ")
#         age_input = input("Enter age: ")
#         processed_input = data_handler.preprocess_data([user_input])
#         predicted_duration = lr_model.predict(processed_input, age_input, gender_input)
#         print(f"Predicted Duration: {predicted_duration[0]} minutes")


# if __name__ == "__main__":
#     main()











#below = other model code?



import pandas as pd
import pickle
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
    data_file = 'EstimateHealthcareAppointmentLengthGivenX-Sheet2.csv'

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
    })

    # Split the dataset
    X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
        processed_sentences, additional_features.values, durations, test_size=0.2, random_state=50)

    # Train and evaluate the model
    model = LinearRegressionModel()
    model.train(X_train_text, X_train_features, y_train)
    model.evaluate(X_test_text, X_test_features, y_test)
    model.save()

    while True:
        user_input = input("Enter an appointment description (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        gender_input = int(input("Enter gender (0 for Female, 1 for Male): "))
        age_input = int(input("Enter age: "))
        familiarity_input = int(input("Enter familiarity (0 for No, 1 for Yes): "))
        processed_input = data_handler.preprocess_data([user_input])
        additional_features_input = pd.DataFrame({
            'gender': [gender_input], 
            'age': [age_input],
            'familiarity': [familiarity_input]
        })
        predicted_duration = model.predict(processed_input, additional_features_input.values)
        print(f"Predicted Duration: {predicted_duration[0]} minutes")


if __name__ == "__main__":
    main()