from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

DATA_DICT = {
    'english': {
        'data_url': 'https://www.clarin.si/repository/xmlui/bitstream/' +
                    'handle/11356/1054/English_Twitter_sentiment.csv?' +
                    'sequence=3&isAllowed=y',
        'file_name': 'english_twitter_sentiment.csv',
        'cleaned_file_name': 'english_cleaned_twitter_sentiment.csv',
        'tweets_file_name': 'english_tweets.txt',
        'merged_file': 'english_merged.csv'
    },
    'german': {
        'data_url': 'https://www.clarin.si/repository/xmlui/bitstream/' +
                    'handle/11356/1054/German_Twitter_sentiment.csv?' +
                    'sequence=2&isAllowed=y',
        'file_name': 'german_twitter_sentiment.csv',
        'cleaned_file_name': 'german_cleaned_twitter_sentiment.csv',
        'tweets_file_name': 'german_tweets.txt',
        'merged_file': 'german_merged.csv'
    },
    'spanish': {
        'data_url': 'https://www.clarin.si/repository/xmlui/bitstream/' +
                    'handle/11356/1054/Spanish_Twitter_sentiment.csv?' +
                    'sequence=13&isAllowed=y',
        'file_name': 'spanish_twitter_sentiment.csv',
        'cleaned_file_name': 'spanish_cleaned_twitter_sentiment.csv',
        'tweets_file_name': 'spanish_tweets.txt',
        'merged_file': 'spanish_merged.csv'
    },
}

LANGUAGES = ['english', 'german', 'spanish']

# Tunable hyper-parameters
LOWERCASE = True
# STOPWORDS_ENGLISH = 'english'
TOKENIZER = word_tokenize  # Documentation goes here
NGRAM_RANGE = (1, 1)
# STEMMING = None
MLMODEL1 = MultinomialNB()
MLMODEL2 = svm.LinearSVC()

# token_pattern - it takes care of ignoring punctuation marks by default

COUNT_VECTORIZER_PARAMS = {
    # 'stop_words': STOPWORDS_ENGLISH,
    'tokenizer': TOKENIZER,
    'analyzer': 'word',
    'lowercase': LOWERCASE,
    'ngram_range': NGRAM_RANGE,
    # 'max_df': 0.9,
    'max_df': 0.8,
    'min_df': 4,
}

PIPELINE_PARAMS = {
    'vect': CountVectorizer(**COUNT_VECTORIZER_PARAMS),
    'tfidf': TfidfTransformer(),
    'clf': MLMODEL2,
}
