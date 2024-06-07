import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
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
        gender = data.iloc[:, 0].values
        age = data.iloc[:, 1].values.astype(int)
        sentences = sentences = data.iloc[:, 2].values
        durations = data.iloc[:, 3].values.astype(int)

        # sentences = data.iloc[:, 0].values
        # # assign appointment length time to int
        # durations = data.iloc[:, 1].values.astype(int)
        return gender, age, sentences, durations

    def preprocess_data(self, sentences):
        processed_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            # break word down to its root word/meaning
            lemmatized = [self.lemmatizer.lemmatize(word.lower())
                          for word in words if word.lower() not in self.stop_words]
            processed_sentences.append(' '.join(lemmatized))
        return processed_sentences


class NaiveBayesModel:
    def __init__(self):
        #instantiate model Class objs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.model = LinearRegression(alpha=0.1)

    def train(self, X_train, y_train):
        # X is matrix of samples to features
        # y is array of samples
        # statistic model to find word relevance
        X_train_counts = self.vectorizer.fit_transform(X_train)
        # fit model with Tfidvectors
        self.model.fit(X_train_counts, y_train)

    def predict(self, descriptions):
        counts = self.vectorizer.transform(descriptions)
        return self.model.predict(counts)

    def evaluate(self, X_test, y_test):
        # returns matrix of n samples, n features
        X_test_counts = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_counts)
        # accuracy = accuracy_score(y_test, predictions)

    def save(self, path='naive_bayes_model.pickle'):
        with open(path, 'wb') as handle:
            pickle.dump((self.vectorizer, self.model), handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    data_file = 'EstimateHealthcareAppointmentLengthGivenX-Sheet1.csv'

    # load and preprocess data
    data_handler = DataHandler(data_file)
    gender, age, sentences, duration = data_handler.load_data()
    processed_sentences = data_handler.preprocess_data(sentences)

    #rearange so splitting rest of variables
    #have a column for gender age age
    #gender convert into 0s and 1s
    #turn sentecnes into matrix where each row is training ex and row is count of number of words in doc
    #append to matrix columns for gender and age

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(processed_sentences, durations, test_size=0.2, random_state=50)

    # train Naive Bayes Model
    nb_model = NaiveBayesModel()
    nb_model.train(X_train, y_train)
    nb_model.evaluate(X_test, y_test)
    nb_model.save()

    while True:
        user_input = input("Enter an appointment description (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        processed_input = data_handler.preprocess_data([user_input])
        predicted_duration = nb_model.predict(processed_input)
        print(f"Predicted Duration: {predicted_duration[0]} minutes")


if __name__ == "__main__":
    main()