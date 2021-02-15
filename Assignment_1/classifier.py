import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from constants import given_data_root_path
from LP_toolkits import tokenize


def load_dataset_basic(csv_file_address):
    """

    :param csv_file_address: csv file address (it could be train, develop or test file)
    :return tweets_data: a dictionary of those needed feature only for basic model training
    """
    csv_data = pd.read_csv(csv_file_address, encoding='utf8')
    tweets_data = dict()
    tweets_data.update({'text': list()})
    tweets_data.update({'airline_sentiment': list()})
    for index in range(csv_data.shape[0]):
        tweet = ' '.join(tokenize(csv_data['text'][index]))
        assigned_class = csv_data['airline_sentiment'][index]

        tweets_data['text'].append(tweet)
        tweets_data['airline_sentiment'].append(assigned_class)

    return tweets_data


def load_dataset_advance(csv_file_address):
    """
        :param csv_file_address: csv file address (it could be train, develop or test file)
        :return tweets_data: a dictionary of those needed feature only for basic model training
        """

    csv_data = pd.read_csv(csv_file_address, encoding='utf8')
    tweets_data = {'text': list(),
                   'airline_sentiment': list(),
                   'airline_sentiment:confidence': list(),
                   'retweet_count': list()}

    for index in range(csv_data.shape[0]):
        tweet = ' '.join(tokenize(csv_data['text'][index]))
        # assigned_class = csv_data['airline_sentiment'][index]

        tweets_data['text'].append(tweet)
        tweets_data['airline_sentiment'].append(csv_data['airline_sentiment'][index])
        tweets_data['airline_sentiment:confidence'].append(csv_data['airline_sentiment:confidence'][index])
        tweets_data['retweet_count'].append(csv_data['retweet_count'][index])

    return tweets_data


#   use for advance tweets data in order to includes 2 other feature than text to classification
def extend_classes(advance_tweets_data_dict):
    """

    :param advance_tweets_data_dict: a dictionary of advanced extracted data
                                    (sth like what load_data_advance function returns)
    :return tweet_reformed_data: a dictionary of tweets 'text' and 'airline_sentiment' which is one of bellow that are
                                extended form

    new airline_sentiments:'high-negative', 'negative', 'low-negative', 'neutral',
                            'low-positive', 'positive', 'high-positive'
    used 'retweet_count' and 'airline_sentiment:confidence' simultaneously
    to assign a rate between 0 to 3 to each tweet and reassign them one of 5 mentioned classes
    """
    tweet_reformed_data = {'text': list(),
                           'airline_sentiment': list()}
    new_classes = {'high-negative', 'negative', 'low-negative', 'neutral',
                   'low-positive', 'positive', 'high-positive'}
    avg_retweet = float(sum(advance_tweets_data_dict['retweet_count'])) / float(len(advance_tweets_data_dict))

    for index in range(len(advance_tweets_data_dict['text'])):
        tweet_reformed_data['text'].append(advance_tweets_data_dict['text'][index])

        if advance_tweets_data_dict['airline_sentiment'][index] == 'neutral':
            tweet_reformed_data['airline_sentiment'].append(advance_tweets_data_dict['airline_sentiment'][index])
        elif advance_tweets_data_dict['airline_sentiment'][index] == 'negative':
            scaled_retweet = int()
            if advance_tweets_data_dict['retweet_count'][index] > avg_retweet:
                scaled_retweet = 3
            else:
                scaled_retweet = 3 * float(advance_tweets_data_dict['retweet_count'][index]) / avg_retweet
            con = float(
                2 * 3 * advance_tweets_data_dict['airline_sentiment:confidence'][index] + scaled_retweet) / 3.
            if con <= 1:
                tweet_reformed_data['airline_sentiment'].append('low-negative')
            elif con <= 2:
                tweet_reformed_data['airline_sentiment'].append('negative')
            elif con <= 3:
                tweet_reformed_data['airline_sentiment'].append('high-negative')
        elif advance_tweets_data_dict['airline_sentiment'][index] == 'positive':
            scaled_retweet = int()
            if advance_tweets_data_dict['retweet_count'][index] > avg_retweet:
                scaled_retweet = 3
            else:
                scaled_retweet = 3 * float(advance_tweets_data_dict['retweet_count'][index]) / avg_retweet
            con = float(
                2 * 3 * advance_tweets_data_dict['airline_sentiment:confidence'][index] + scaled_retweet) / 3.
            if con <= 1:
                tweet_reformed_data['airline_sentiment'].append('low-positive')
            elif con <= 2:
                tweet_reformed_data['airline_sentiment'].append('positive')
            elif con <= 3:
                tweet_reformed_data['airline_sentiment'].append('high-positive')

    # with open('adv_data_reformed.pickle', 'wb') as ad_file:
    #     pickle.dump(tweet_reformed_data, ad_file, protocol=pickle.HIGHEST_PROTOCOL)

    return tweet_reformed_data


def chi_square_calculator(tweets_data_dic):
    """

    :param tweets_data_dic: a dictionary of tweets data that only have 'text' and 'airline_sentiment' keys
    :return chi_square_table: an np.array object that cell ij contains chi_square value for term j and class i
    """
    index = 0
    classes_index = dict()
    for c in np.unique(np.array(list(tweets_data_dic['airline_sentiment']))):
        classes_index.update({c: index})
        index += 1
    terms_index = dict()
    index = 0

    # code each term in order to construct a contingency table to use it for CHI 2 calculation
    for tweet in tweets_data_dic['text']:
        tweet = tweet.split()
        for term in tweet:
            if term not in terms_index.keys():
                terms_index.update({term: index})
                index += 1

    contingency_table = np.zeros((len(classes_index), len(terms_index)), dtype=float)
    for index in range(len(tweets_data_dic['text'])):
        c = tweets_data_dic['airline_sentiment'][index]
        tweet = tweets_data_dic['text'][index].split()
        for term in tweet:
            contingency_table[classes_index[c]][terms_index[term]] += 1

    # calculate CHI square for term k in class c
    chi_square_table = np.zeros((len(classes_index), len(terms_index)), dtype=float)

    total_counts = sum(sum(contingency_table))
    for t_index in terms_index.values():
        for c_index in classes_index.values():
            observed = contingency_table[c_index][t_index]
            expected = sum(contingency_table[c_index, :]) * sum(
                contingency_table[:, t_index]) / total_counts
            chi_square_table[c_index][t_index] = pow(observed - expected, 2) / expected
    return chi_square_table, terms_index, classes_index


def vectorize(features_dict, text_arr):
    """
    :param features_dict: a dictionary of indexed features that have been extracted based on chi values and
                            use them as a vocabulary for vectorizing
    :param text_arr: a list of  tweets strings
    :return vector: a matrix of vectorized tweets
    """
    vectorizer = CountVectorizer(vocabulary=features_dict)
    vector = vectorizer.fit_transform(text_arr)

    return vector


def train_model(features_dict, tweets_data_dict, advance):
    """
    :param features_dict: a dictionary of indexed features
    :param tweets_data_dict:  a dictionary of tweets data that only have 'text' and 'airline_sentiment' keys
    :param advance:
    :return:

    save 'svm_advance_model.pickle' for advance form and save 'svm_basic_model.pickle' for basic form
    in order to use them for prediction and evaluation
    """
    if advance:
        y_vector = tweets_data_dict['airline_sentiment']
        x_vector = vectorize(features_dict, tweets_data_dict['text'])

        clf = SVC(kernel='linear', C=1, decision_function_shape='ovr')
        clf.fit(x_vector, y_vector)
        return clf
    else:

        y_vector = tweets_data_dict['airline_sentiment']
        x_vector = vectorize(features_dict, tweets_data_dict['text'])

        clf = SVC(kernel='linear', C=1, decision_function_shape='ovr')
        clf.fit(x_vector, y_vector)

        return clf


def basic_classifier_training(train_csv_file_address, develop_csv_file_address):
    """
    :param develop_csv_file_address: dev_set csv file address
    :param train_csv_file_address: train_set csv file address
    :return:
    save the model in svm_basic_model.pickle file and indexed features dictionary in features_basic.pickle file
     for further usage
    """
    tweets_data_dic = load_dataset_basic(train_csv_file_address)
    print('calculate chi square ')
    chi_square_matrix, terms_index, classes_index = chi_square_calculator(tweets_data_dic)
    print('extract feature')

    chi_square_sorted = sorted(chi_square_matrix.reshape(chi_square_matrix.shape[0] * chi_square_matrix.shape[1]),
                               reverse=True)

    print('model training')
    score = 0.
    dev_data = load_dataset_basic(develop_csv_file_address)
    # split chi_square_sorted i to 20 chunks and use airline-dev.csv decide about features sets
    for iteration in range(20):
        limit = chi_square_sorted[int((iteration + 1) * len(chi_square_sorted) / 20) - 1]
        index = 0
        features = dict()
        for term in terms_index.keys():
            for c in classes_index.keys():
                if chi_square_matrix[classes_index[c]][terms_index[term]] >= limit:
                    if term not in features.keys():
                        features.update({term: index})
                        index += 1


        model = train_model(features_dict=features, tweets_data_dict=tweets_data_dic, advance=False)
        m_score = model.score(vectorize(features, dev_data['text']), dev_data['airline_sentiment'])
        if score < m_score:
            score = m_score
            # save model and features set for the highest score
            with open('svm_basic_model.pickle', 'wb') as svm_file:
                pickle.dump(model, svm_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open('features_basic.pickle', 'wb') as feature_list:
                pickle.dump(features, feature_list, protocol=pickle.HIGHEST_PROTOCOL)


def advance_classifier_training(train_csv_file_address, develop_csv_file_address):
    """

    :param develop_csv_file_address: dev_set csv file address
    :param train_csv_file_address: train_set csv file address
    :return:
    save the model in svm_advance_model.pickle file and indexed features dictionary in features_advance.pickle file
     for further usage
    """
    tweets_data_dic = extend_classes(load_dataset_advance(train_csv_file_address))
    print('calculate chi square ')
    chi_square_matrix, terms_index, classes_index = chi_square_calculator(tweets_data_dic)
    print('extract feature')

    chi_square_sorted = sorted(chi_square_matrix.reshape(chi_square_matrix.shape[0] * chi_square_matrix.shape[1]),
                               reverse=True)

    print('model training')
    score = 0.
    dev_data = extend_classes(load_dataset_advance(develop_csv_file_address))
    # split chi_square_sorted i to 20 chunks and use airline-dev.csv decide about features sets
    for iteration in range(20):
        limit = chi_square_sorted[int((iteration + 1) * len(chi_square_sorted) / 20) - 1]
        index = 0
        features = dict()
        for term in terms_index.keys():
            for c in classes_index.keys():
                if chi_square_matrix[classes_index[c]][terms_index[term]] >= limit:
                    if term not in features.keys():
                        features.update({term: index})
                        index += 1


        model = train_model(features_dict=features, tweets_data_dict=tweets_data_dic, advance=False)
        m_score = model.score(vectorize(features, dev_data['text']), dev_data['airline_sentiment'])

        if score < m_score:
            score = m_score
            # save model and features set for the highest score
            with open('svm_advance_model.pickle', 'wb') as svm_file:
                pickle.dump(model, svm_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open('features_advance.pickle', 'wb') as feature_list:
                pickle.dump(features, feature_list, protocol=pickle.HIGHEST_PROTOCOL)


def evaluation(test_data_file_add, advance):
    """

    :param test_data_file_add: test file (a string ot it's address with file name)
    :param advance: if the features have been  extracted by advanced method or not
    :return:

    print evaluation metrics based on the given test set.
    """

    true_y = list()
    y_pred = list()
    if advance:
        print('********************************     ADVANCE     ********************************')

        test_data = extend_classes(load_dataset_advance(test_data_file_add))
        test_x = list()
        true_y = test_data['airline_sentiment']
        # vectorize tweets text in order to evaluate model
        with open('features_advance.pickle', 'rb') as feature_file:
            features = pickle.load(feature_file, encoding='utf8')
            test_x = vectorize(features, test_data['text'])

        with open('svm_advance_model.pickle', 'rb') as SVM_model_file:
            model = pickle.load(SVM_model_file)
            y_pred = model.predict(test_x)
    else:
        print('********************************      BASIC      ********************************')

        test_data = load_dataset_basic(test_data_file_add)
        test_x = list()
        true_y = test_data['airline_sentiment']
        # vectorize tweets text in order to evaluate model
        with open('features_basic.pickle', 'rb') as feature_file:
            features = pickle.load(feature_file, encoding='utf8')
            test_x = vectorize(features, test_data['text'])

        with open('svm_basic_model.pickle', 'rb') as SVM_model_file:
            model = pickle.load(SVM_model_file)
            y_pred = model.predict(test_x)

    # index classes
    index = 0
    class_index = dict()
    for c in np.unique(np.array(true_y)):
        class_index.update({c: index})
        index += 1

    #     construct confusion matrix or ech class based on the order they've been indexed
    #     its general and will work for more than three classes
    confusion_matrix = np.zeros((len(class_index), 2, 2), dtype=int)
    for i in range(len(true_y)):
        if true_y[i] == y_pred[i]:
            # True Positive
            confusion_matrix[class_index[true_y[i]]][1][1] += 1
            # True Negative for other 2
            for c in class_index.keys():
                if c != true_y[i]:
                    confusion_matrix[class_index[c]][0][0] += 1
        else:
            # False Positive:
            confusion_matrix[class_index[true_y[i]]][0][1] += 1
            # False Negative
            confusion_matrix[class_index[y_pred[i]]][1][0] += 1
            # True Negative for the other class
            for c in class_index.keys():
                if c != true_y[i] and c != y_pred[i]:
                    confusion_matrix[class_index[c]][0][0] += 1
    """
    TN  FP
    FN  TP
    macro: avg
    micro: calculate sum
    """

    print('class indexes: {}'.format(class_index))
    print('confusion_matrix:    \n{}'.format(confusion_matrix))

    # calculate precision for each class using confusion matrix
    precision_per_class = dict()
    for c in class_index.keys():
        if (confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][0][1]) != 0:
            p = float(confusion_matrix[class_index[c]][1][1]) / float(
                confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][0][1])
        else:
            p = 0
        precision_per_class.update({c: p})

    # calculate recall for each class using confusion matrix
    recall_per_class = dict()
    for c in class_index.keys():
        if (confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][1][0]) != 0:
            r = float(confusion_matrix[class_index[c]][1][1]) / float(
                confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][1][0])
        else:
            r = 0
        recall_per_class.update({c: r})

    # calculate f1 for each class using confusion matrix
    f1_per_class = dict()
    for c in class_index.keys():
        if (2 * confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][1][0] +
            confusion_matrix[class_index[c]][0][1]) != 0:
            f1 = 2 * float(confusion_matrix[class_index[c]][1][1]) / float(
                2 * confusion_matrix[class_index[c]][1][1] + confusion_matrix[class_index[c]][1][0] +
                confusion_matrix[class_index[c]][0][1])
        else:
            f1 = 0
        f1_per_class.update({c: f1})

    # calculate error rate for each class using confusion matrix
    error_rate_per_class = dict()
    for c in class_index.keys():

        error_rate = float(confusion_matrix[class_index[c]][0][1]+confusion_matrix[class_index[c]][1][0]) / \
                     float(sum(sum(confusion_matrix[class_index[c]])))
        error_rate_per_class.update({c: error_rate})

    total = [[0, 0], [0, 0]]
    # calculate total TP, TN, FP and FN
    for c in class_index.keys():
        total[0][0] += confusion_matrix[class_index[c]][0][0]
        total[0][1] += confusion_matrix[class_index[c]][0][1]
        total[1][0] += confusion_matrix[class_index[c]][1][0]
        total[1][1] += confusion_matrix[class_index[c]][1][1]

    accuracy = float()
    if (total[0][0] + total[0][1] + total[1][0] + total[1][1]) != 0:
        accuracy = float(total[0][0] + total[1][1]) / float(total[0][0] + total[0][1] + total[1][0] + total[1][1])
    else:
        accuracy = 0
    f1_micro = float()
    if float(2 * total[1][1] + total[1][0] + total[0][1]) != 0:
        f1_micro = 2 * float(total[1][1]) / float(2 * total[1][1] + total[1][0] + total[0][1])
    else:
        f1_micro = 0
    f1_macro = float(sum(f1_per_class.values())) / float(len(class_index))

    print('precision per class:     {}'.format(precision_per_class))
    print('recall per class:        {}'.format(recall_per_class))
    print('f1 per class             {}'.format(f1_per_class))
    print('error rate per class     {}\n'.format(error_rate_per_class))

    print('avg precision:           {}'.format(sum(precision_per_class.values()) / len(class_index)))
    print('accuracy:                {}'.format(accuracy))
    print('f1 macro:                {}'.format(f1_macro))
    print('f1 micro:                {}'.format(f1_micro))


def predict(tweet, advance):
    """

    :param tweet: a string which contains tweet text
    :param advance: True if its advance mode, False ow.
    :return: an assigned class
    """
    y_pred = list()
    if advance:
        features = dict()
        with open('features_advance.pickle', 'rb') as features_file:
            features = pickle.load(features_file, encoding='utf8')

            tweet_list = list()
            tweet_list.append(' '.join(tokenize(tweet)))
            vector_tweet = vectorize(features, tweet_list)
            with open('svm_advance_model.pickle', 'rb') as svm_model_file:
                model = pickle.load(svm_model_file)
                y_pred = model.predict(vector_tweet)
    else:
        features = dict()
        with open('features_basic.pickle', 'rb') as features_file:
            features = pickle.load(features_file, encoding='utf8')

            tweet_list = list()
            tweet_list.append(' '.join(tokenize(tweet)))
            vector_tweet = vectorize(features, tweet_list)
            with open('svm_basic_model.pickle', 'rb') as svm_model_file:
                model = pickle.load(svm_model_file)
                y_pred = model.predict(vector_tweet)

    return y_pred


if __name__ == '__main__':
    print('classifier')
    # basic_classifier_training(given_data_root_path + 'airline-train.csv', given_data_root_path + 'airline-dev.csv')
    # advance_classifier_training(given_data_root_path + 'airline-train.csv', given_data_root_path + 'airline-dev.csv')

    # evaluation(given_data_root_path + 'airline-test.csv', advance=False)
