import pandas as pd
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
        sentences = data.iloc[:, 0].values
        durations = data.iloc[:, 1].values.astype(int)
        return sentences, durations

    def preprocess_data(self, sentences):
        processed_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            lemmatized = [self.lemmatizer.lemmatize(word.lower())
                          for word in words if word.lower() not in self.stop_words]
            processed_sentences.append(' '.join(lemmatized))
        return processed_sentences

class NaiveBayesModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.model = MultinomialNB(alpha=0.1)

    def train(self, X_train, y_train):
        X_train_counts = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_counts, y_train)

    def predict(self, descriptions):
        counts = self.vectorizer.transform(descriptions)
        return self.model.predict(counts)

    def evaluate(self, X_test, y_test):
        X_test_counts = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_counts)

    def save(self, path='naive_bayes_model.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    data_file = 'EstimateHealthcareAppointmentLengthGivenX-Sheet1.csv'

    # Load and preprocess data
    data_handler = DataHandler(data_file)
    sentences, durations = data_handler.load_data()
    processed_sentences = data_handler.preprocess_data(sentences)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(processed_sentences, durations, test_size=0.2, random_state=50)

    # Train Naive Bayes Model
    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train)
    nb_model.evaluate(X_test, y_test)
    nb_model.save()

    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        processed_input = data_handler.preprocess_data([user_input])
        predicted_duration = nb_model.predict(processed_input)
        print(predicted_duration[0])

if __name__ == "__main__":
    main()