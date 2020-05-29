import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    """Load the filepath and return the data"""
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine) # is table always called this? 
    print(df.head())
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Tokenize text (a disaster message).
    Args:
        text: String. A disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list. It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_model():
    """Build model.
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                learning_rate=0.3,
                n_estimators=200
            )
        ))
    ])

    # Set parameters for gird search
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv


def save_model(model, model_filepath):
    """Save stats
    Args;
        X: numpy.ndarray. Disaster messages.
        Y: numpy.ndarray. Disaster categories for each messages.
        category_names: Disaster category names.
        vocaburary_stats_filepath: String. Vocaburary stats is saved as pickel into this file.
        category_stats_filepath: String. Category stats is saved as pickel into this file.
    """
    # Check vocabulary
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X)

    # Convert vocabulary into pandas.dataframe
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocabulary_df = pd.DataFrame.from_dict({'words': keys, 'counts': values})

    # Vocabulary stats
    vocabulary_df = vocabulary_df.sample(30, random_state=72).sort_values('counts', ascending=False)
    vocabulary_counts = list(vocabulary_df['counts'])
    vocabulary_words = list(vocabulary_df['words'])

    # Save vocaburaly stats
    with open(vocabulary_stats_filepath, 'wb') as vocabulary_stats_file:
        pickle.dump((vocabulary_counts, vocabulary_words), vocabulary_stats_file)

    # Category stats
    category_counts = list(Y.sum(axis=0))

    # Save category stats
    with open(category_stats_filepath, 'wb') as category_stats_file:
        pickle.dump((category_counts, list(category_names)), category_stats_file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()