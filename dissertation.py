import numpy as np
import pandas as pd
import re

from collections import Counter

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline


from config import DATA_DICT, PIPELINE_PARAMS
from fetch_tweets import main as fetch_tweets_main


def download_data_files(languages):
    """
    Downloads the data files for the given languages.

    Input -
        languages - list of str, languages for which we want to
            download the data.

    Returns -
        None.
    """
    for language in languages:
        data_url = DATA_DICT[language]['data_url']
        file_name = DATA_DICT[language]['file_name']
        df = pd.read_csv(data_url)
        df.to_csv(file_name)


def remove_duplicates(languages):
    """
    Removes the duplicate tweet ids from the respective CSV files.

    Input -
        languages - list of str, languages for which we want to remove
            duplicates.

    Returns -
        None.
    """
    def get_cleaned_df(df):
        """
        Gets a dataframe that may have duplicate tweet ids and return
        one without any duplicates.

        Input -
            df - DataFrame, the columns are TweetID, HandLabel,
                AnnotatorID.

        Returns -
            cleaned_df - the dataframe, with all the conflicts
                resolved.
        """
        tweet_ids = df['TweetID']
        tweet_id_count = Counter(tweet_ids)

        duplicate_tweet_ids = [tweet_id for tweet_id in tweet_id_count.keys()
                               if tweet_id_count[tweet_id] > 1]
        cleaned_df = df.copy()

        # Loop checking for each duplicate id
        for tweet_id in duplicate_tweet_ids:
            df_dup = df[df['TweetID'] == tweet_id]
            labels = set(df_dup['HandLabel'])

            # Removing or replacing duplicates
            if len(labels) == 1:
                # Only one hand label, remove all but one rows
                duplicate_indices = df_dup.index[1:]
                # Getting the excess rows with the duplicate label
                cleaned_df = cleaned_df.drop(duplicate_indices)
            elif len(labels) == 2:
                # We have to check for specific cases now
                if 'neutral' in labels:
                    neutral_indices = df_dup[df_dup['HandLabel'] ==
                                             'neutral'].index
                    # Getting the indices of all neutral rows
                    duplicate_indices = df_dup[df_dup['HandLabel'] !=
                                               'neutral'].index[1:]
                    # Getting the excess rows with the extreme label
                    cleaned_df = cleaned_df.drop(neutral_indices)
                    cleaned_df = cleaned_df.drop(duplicate_indices)
                else:
                    positive_row_count = len(df_dup[df_dup['HandLabel'] ==
                                                    'positive'])
                    negative_row_count = len(df_dup[df_dup['HandLabel'] ==
                                                    'negative'])
                    final_label = None
                    if positive_row_count > negative_row_count:
                        final_label = 'positive'
                    elif positive_row_count < negative_row_count:
                        final_label = 'negative'
                    if final_label:  # final_label has a non-empty value in it,
                                        # either positive or negative
                        minority_indices = df_dup[df_dup['HandLabel'] !=
                                                  final_label].index
                        duplicate_indices = df_dup[df_dup['HandLabel'] ==
                                                   final_label].index[1:]
                        cleaned_df = cleaned_df.drop(minority_indices)
                        cleaned_df = cleaned_df.drop(duplicate_indices)
                    else:       # When negative and positive row count is equal
                        cleaned_df = \
                            cleaned_df[cleaned_df['TweetID'] != tweet_id]
            else:
                # Remove all the rows. Or check for majority. Again the same
                # assumption.
                positive_row_count = len(df_dup[df_dup['HandLabel'] ==
                                                'positive'])
                negative_row_count = len(df_dup[df_dup['HandLabel'] ==
                                                'negative'])
                neutral_row_count = len(df_dup[df_dup['HandLabel'] ==
                                               'neutral'])

                final_label = None
                if positive_row_count > negative_row_count and \
                        positive_row_count > neutral_row_count:
                    final_label = 'positive'
                elif negative_row_count > positive_row_count and \
                        negative_row_count > neutral_row_count:
                    final_label = 'negative'
                elif neutral_row_count > positive_row_count and \
                        neutral_row_count > negative_row_count:
                    final_label = 'neutral'
                if final_label:
                    # final_label has a non-empty value in it, either positive
                    # or negative
                    minority_indices = df_dup[df_dup['HandLabel'] !=
                                              final_label].index
                    duplicate_indices = df_dup[df_dup['HandLabel'] ==
                                               final_label].index[1:]
                    cleaned_df = cleaned_df.drop(minority_indices)
                    cleaned_df = cleaned_df.drop(duplicate_indices)
                else:  # When negative and positive row count is equal
                    cleaned_df = cleaned_df[cleaned_df['TweetID'] != tweet_id]
        return cleaned_df

    for language in languages:
        file_name = DATA_DICT[language]['file_name']
        cleaned_file_name = DATA_DICT[language]['cleaned_file_name']
        df = pd.read_csv(file_name)
        cleaned_df = get_cleaned_df(df)
        cleaned_df.to_csv(cleaned_file_name)


def fetch_tweets(languages):
    """
    Fetch tweets for the languages given and save them in the
    respective files.

    Input -
        languages - list of str, languages for which we want to fetch
            tweets.

    Returns -
        None.
    """
    for language in languages:
        cleaned_file_name = DATA_DICT[language]['cleaned_file_name']
        tweets_file_name = DATA_DICT[language]['tweets_file_name']
        args = [cleaned_file_name, tweets_file_name]
        fetch_tweets_main(args)


def merge_dfs(languages):
    """
    Inputs -
        languages - list of str, languages for which we want to merge
            the labels and the tweets dataframes.

    Working-
        1. Reads both the files using pandas.read_csv().
        2. Merges the dataframes on TweetID.

    Returns -
        merged_df - The merged dataframe.
    """
    for language in languages:
        cleaned_labels_file = DATA_DICT[language]['cleaned_file_name']
        fetched_tweets_file = DATA_DICT[language]['tweets_file_name']
        merged_file = DATA_DICT[language]['merged_file']
        cleaned_labels_df = pd.read_csv(cleaned_labels_file)
        tweets_df = pd.read_csv(fetched_tweets_file)
        merged_df = pd.merge(cleaned_labels_df, tweets_df, how='inner',
                             on='TweetID', suffixes=('_x', '_y'), copy=True,
                             validate=None)
        tweet_texts = merged_df['tweet_texts']
        tweet_texts = [re.sub('@[^\s]+', '', text) for text in tweet_texts]
        tweet_texts = [re.sub('http\S+', '', text) for text in tweet_texts]
        merged_df['tweet_texts'] = tweet_texts
        merged_df.to_csv(merged_file)


def predict_sentiments(language, pipeline_params=PIPELINE_PARAMS, k=10):
    """
    Inputs -
        language - language for which we are predicting the sentiments.
        k - number of folds to split the data into, while performing
            k-fold cross-validation.

    Working -
        1. Get the relevant merged dataframe.
        2. Extract the predictors (tweet_texts) and targets
            (HandLabel).
        3. Split predictors and targets into train and test datasets.
        4. Build a Pipeline with the required operations.
        5. Get the cross-validation scores after performing k-fold
            cross-validation on the training dataset using the
            Pipeline.
        6. Train the Pipeline using the entire training data.
        7. Predict for the test data.
        8. Get the classification report for the Pipeline on the test
            data.

    Returns -
        cv_scores - The cross-validation scores.
        classification_report - Classification report on the test data.
    """
    # Step 1
    merged_file = DATA_DICT[language]['merged_file']
    merged_df = pd.read_csv(merged_file)

    # Step 2
    df_predictor = merged_df['tweet_texts']
    df_target = merged_df['HandLabel']

    # Step 3
    df_predictor_train, df_predictor_test, df_target_train, df_target_test = \
        train_test_split(df_predictor, df_target, test_size=0.20,
                         random_state=40, shuffle=True, stratify=df_target)

    # Step 4
    pipeline_params = [(key, value) for key, value in pipeline_params.items()]
    text_clf = Pipeline(pipeline_params)

    # Step 5
    cv_scores = cross_val_score(text_clf, df_predictor_train,
                                df_target_train, cv=k)

    # Steps 6 & 7
    text_clf.fit(df_predictor_train, df_target_train)
    predicted = text_clf.predict(df_predictor_test)

    # Step 8
    accuracy = np.mean(predicted == df_target_test)
    classification_report = metrics.classification_report(df_target_test,
                                                          predicted, )
    return cv_scores, accuracy, classification_report
