import pickle
from parsivar import FindStems
from hazm import stopwords_list, Lemmatizer
import numpy as np

from constants import given_doc_root_path, document_root_path, limit_index
from LP_toolkits import normalizer

Lemmatizer = Lemmatizer()
Stemmer = FindStems()
stopwords = set(stopwords_list())


# define stemmer function.
def stemmer(email):
    """
    :param email: a string of email text
    :return: a string of input in which for each verb it's root has been replaced
    """
    tokens = ''
    for word in email.split():
        token = Lemmatizer.lemmatize(word)
        if '#' in token:
            token = token.split('#')
            if word in token[0]:
                token = token[0]
            else:
                token = token[1]
        else:
            token = Stemmer.convert_to_stem(word)
            if '&' in token:
                token = token.split('&')
                if word in token[0]:
                    token = token[0]
                else:
                    token = token[1]

        tokens += token + ' '

    return tokens


# Normalizing and Stemming Data
#
# read all training data and use a dictionary structure like:
# train= {'ham': list('normalized and stemmed news'), 'spam': list('normalized and stemmed news')}


def prepare_data(ham_dir, spam_dir, train=True):
    """

    :param ham_dir: path to where all ham's email text files are in
    :param spam_dir: path to where all spam's email text files are in
    :param train: True if using this function for training data, False if using this function for testing data
    :return: processed_dict= {'ham': list('normalized and stemmed news'), 'spam': list('normalized and stemmed news')}
    """
    doc_id = 1
    # contains processed documents
    processed_dict = {'ham': list(),
                      'spam': list()}
    ham_file = ''
    spam_file = ''
    if train:
        ham_file = 'hamtraining ({}).txt'
        spam_file = 'spamtraining ({}).txt'
    else:
        ham_file = 'hamtesting ({}).txt'
        spam_file = 'spamtesting ({}).txt'
    while True:
        try:
            with open(ham_dir + ham_file.format(doc_id), 'r', encoding='utf8') as train_file:
                email = stemmer(normalizer(train_file.read()))
                # remove stopwords
                email_tokens = email.split()
                email = ''
                for token in email_tokens:
                    if token not in stopwords:
                        email += token + ' '
                processed_dict['ham'].append(email)
                doc_id += 1
        except IOError:
            break
    doc_id = 1
    while True:
        try:
            with open(spam_dir + spam_file.format(doc_id), 'r', encoding='utf8') as train_file:
                email = stemmer(normalizer(train_file.read()))
                # remove stopwords
                email_tokens = email.split()
                email = ''
                for token in email_tokens:
                    if token not in stopwords:
                        email += token + ' '
                processed_dict['spam'].append(email)
                doc_id += 1
        except IOError:
            break
    return processed_dict


# CHI square calculator
def chi_square_calculator(processed_emails_dict):
    """

    :param processed_emails_dict: a dictionary of email like prepare_data's output
    :return chi_square_table: an np.array object that cell ij contains chi_square value for term j and class i
                                ,classes_index and terms_index
    """

    classes_index = {class_name: index for index, class_name in enumerate(processed_emails_dict.keys())}
    terms_index = dict()

    index = 0
    # code each term in order to construct a contingency table to use it for CHI 2 calculation
    for email in processed_emails_dict['ham']:
        email = email.split()
        for term in email:
            if term not in terms_index.keys():
                terms_index.update({term: index})
                index += 1
    for email in processed_emails_dict['spam']:
        email = email.split()
        for term in email:
            if term not in terms_index.keys():
                terms_index.update({term: index})
                index += 1

    contingency_table = np.zeros((len(classes_index), len(terms_index)), dtype=float)

    # first update contingency matrix for emails in ham class
    for index, email in enumerate(processed_emails_dict['ham']):
        email_terms = email.split()
        for term in email_terms:
            contingency_table[classes_index['ham']][terms_index[term]] += 1

    # then update contingency matrix for emails in spam class
    for index, email in enumerate(processed_emails_dict['spam']):
        email_terms = email.split()
        for term in email_terms:
            contingency_table[classes_index['spam']][terms_index[term]] += 1
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


# find most important tokens using chi square
def most_important_tokens(processed_emails_dict):
    """

    :param processed_emails_dict: a dictionary like prepared_data's output
    :return: a dictionary of indexed most important words
    """
    chi_square_matrix, terms_index, classes_index = chi_square_calculator(processed_emails_dict)
    # sort chi square matrix in order to get 500 highest values
    chi_square_sorted = sorted(chi_square_matrix.reshape(chi_square_matrix.shape[0] * chi_square_matrix.shape[1]),
                               reverse=True)
    prev_limit = 0
    index = 0
    most_frequent_terms = dict()
    for limit in chi_square_sorted:
        for term in terms_index.keys():
            for c in classes_index.keys():
                if limit <= chi_square_matrix[classes_index[c]][terms_index[term]] < prev_limit:
                    if term not in most_frequent_terms:
                        most_frequent_terms.update({term: index})
                        index += 1
                        if index >= limit_index:
                            break
            if index >= limit_index:
                break
        if index >= limit_index:
            break
        prev_limit = limit
    return most_frequent_terms


# Vocabulary constructor
def construct_vocabulary(processed_emails_dict):
    """
        :param processed_emails_dict: a dictionary same as prepare_train_data function's output

        save both type of vocabulary on disk for further usage
    """
    emails = list()
    emails.extend(processed_emails_dict['ham'])
    emails.extend(processed_emails_dict['spam'])

    vocabulary = dict()
    index = 0
    for email in emails:
        for term in email.split():
            if term not in vocabulary.keys():
                vocabulary.update({term: index})
                index += 1
    # for using all tokens as a word
    with open(document_root_path + 'vocabulary_simple.pickle', 'wb') as vocabulary_file:
        pickle.dump(vocabulary, vocabulary_file, protocol=pickle.HIGHEST_PROTOCOL)

    vocabulary = most_important_tokens(processed_emails_dict)
    with open(document_root_path + 'vocabulary_advance.pickle', 'wb') as vocabulary_file:
        pickle.dump(vocabulary, vocabulary_file, protocol=pickle.HIGHEST_PROTOCOL)


# vectorize Emails
def vectorize_email(processed_email, vocabulary):
    """

    :param processed_email: a string of processed email
    :param vocabulary: a dictionary in which each key is a term and it's value is that term's assigned index
    :return:
    """
    vec = np.zeros((len(vocabulary)), dtype=int)
    for term in processed_email.split():
        if term in vocabulary.keys():
            vec[vocabulary[term]] += 1
    return vec


def vectorize_data(processed_emails_dict, vocabulary):
    """
    :param processed_emails_dict: a dictionary like prepare_train_data function's output
    :param vocabulary: a dictionary in which each key is a term and it's value is that's term index
    :return a dictionary of vectorized emails like vectorized_emails
    """
    vectorized_emails = {'ham': list(),
                         'spam': list()}

    for email in processed_emails_dict['ham']:
        vectorized_emails['ham'].append(vectorize_email(email, vocabulary))

    for email in processed_emails_dict['spam']:
        vectorized_emails['spam'].append(vectorize_email(email, vocabulary))

    return vectorized_emails


if __name__ == '__main__':
    print(stemmer('می‌گفتم'))