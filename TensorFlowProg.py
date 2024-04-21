import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
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
            lemmatized = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in self.stop_words]
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
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')
        print(classification_report(y_test, predictions))

    def save(self, path='naive_bayes_model.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    data_file = 'Estimate Healthcare Appointment Length Given X - Sheet1.csv'

    # Load and preprocess data
    data_handler = DataHandler(data_file)
    sentences, durations = data_handler.load_data()
    processed_sentences = data_handler.preprocess_data(sentences)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(processed_sentences, durations, test_size=0.2, random_state=50)

    # Training Naive Bayes Model
    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train)
    nb_model.evaluate(X_test, y_test)
    nb_model.save()

    # Example prediction
    test_description = ["Routine dermatology visit for skin care"]
    predicted_duration = nb_model.predict(test_description)
    print(f"Predicted Duration: {predicted_duration[0]} minutes")

    # Grid Search for parameter tuning
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    params = {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],  # unigrams and bigrams
        'multinomialnb__alpha': [0.01, 0.1, 0.5, 1]  # different alpha values
    }
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy')
    grid_search.fit(processed_sentences, durations)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Assuming 'X' and 'y' are your features and labels respectively
    skf = StratifiedKFold(n_splits=5)
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True), MultinomialNB(alpha=0.01))

    # Calculate different scores during cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
    print("Weighted F1-score: {:.2f}".format(np.mean(scores)))

if __name__ == "__main__":
    main()
