from numpy import linalg as la
from numpy import dot
import numpy as np
import pickle
from heapq import nlargest

from preprocessing import prepare_data, vectorize_data
from constants import given_doc_root_path, document_root_path


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (la.norm(vec1) + la.norm(vec2))


def idf_calculator(vectorized_train_dict):
    # store all emails in a list inorder to calculate tf
    emails = list()
    emails.extend(vectorized_train_dict['ham'])
    emails.extend(vectorized_train_dict['spam'])
    emails = np.array(emails)

    # first construct a tf for each index
    n = len(emails)
    vocab_len = len(emails[0])
    idf = [np.log10(n / np.count_nonzero(emails[:, index])) for index in range(vocab_len)]
    return idf


def tf_calculator(vectorized_train_dict):
    tf_train_dict = {'ham': list(),
                     'spam': list()}
    for vec in vectorized_train_dict['ham']:
        tf_vec = np.zeros(vec.shape)
        for index in range(len(vec)):
            tf_vec[index] = np.log10(vec[index] + 1)
        tf_train_dict['ham'].append(tf_vec)

    for vec in vectorized_train_dict['spam']:
        tf_vec = np.zeros(vec.shape)
        for index in range(len(vec)):
            tf_vec[index] = np.log10(vec[index] + 1)
        tf_train_dict['spam'].append(tf_vec)

    return tf_train_dict


def tf_idf_calculator(non_zero_indices, tf, idf):
    score = 0
    for index in non_zero_indices:
        score += tf[index] * idf[index]
    return score


def knn(new_email_vec, similarity_func, vectorized_train_dict=None, idf=None, tf_dict=None):
    """

    :param similarity_func: 0 if using cosine 1 if using tf-idf
    :param new_email_vec: a new vectorized email
    :param vectorized_train_dict: a dictionary like vectorize function output in preprocessing.py file
    if using cosine similarity, None if using tf-idf
    :param idf: None if using cosine idf list if using tf-idf
    :param tf_dict: None if using cosine idf list if using tf-idf

    :return: 'ham' or 'spam'
    """
    k = 7
    indexed_class = dict()
    indexed_scores = dict()

    if similarity_func == 0:

        index = 0
        for vec1 in vectorized_train_dict['ham']:
            indexed_scores.update({index: cosine_similarity(vec1, new_email_vec)})
            indexed_class.update({index: 'ham'})
            index += 1
        for vec1 in vectorized_train_dict['spam']:
            indexed_scores.update({index: cosine_similarity(vec1, new_email_vec)})
            indexed_class.update({index: 'spam'})
            index += 1

    else:
        non_zer_indices = np.nonzero(new_email_vec)[0]

        index = 0
        # calculate tf_idf score for each pair of emails
        for tf in tf_dict['ham']:
            indexed_scores.update({index: tf_idf_calculator(non_zer_indices, tf, idf)})
            indexed_class.update({index: 'ham'})
            index += 1

        for tf in tf_dict['spam']:
            indexed_scores.update({index: tf_idf_calculator(non_zer_indices, tf, idf)})
            indexed_class.update({index: 'spam'})
            index += 1

    # find top k
    top_k_id = nlargest(k, indexed_scores, key=indexed_scores.get)
    count = {'ham': 0, 'spam': 0}
    for index in top_k_id:
        count[indexed_class[index]] += 1
    if count['ham'] > count['spam']:
        return 'ham'
    else:
        return 'spam'


def all_probabilities_calculator(vectorized_train_dict):
    """
    :param vectorized_train_dict: a dictionary like vectorize function output in preprocessing.py file
    :return: all_prob which all_prob['ham'][i] contains p('ham'|wi)  value
    """
    n = len(vectorized_train_dict['ham'][0])
    all_prob = {'ham': np.zeros(n),
                'spam': np.zeros(n)}

    # make them np.array to b able to use its functions
    vectors = {'ham': np.array(vectorized_train_dict['ham']),
               'spam': np.array(vectorized_train_dict['spam'])}

    f_ham = 0
    for index in range(n):
        if sum(vectors['ham'][:, index]) > 0:
            f_ham += 1
    f_spam = 0
    for index in range(n):
        if sum(vectors['spam'][:, index]) > 0:
            f_spam += 1
    # p(ham|wi) = c(wi,'ham')/c(wi)

    total_ham_term = sum([sum(vec) for vec in vectors['ham']])
    for index in range(n):
        all_prob['ham'][index] = ((1 + sum(vectors['ham'][:, index])) / (f_ham + total_ham_term))

    total_spam_term = sum([sum(vec) for vec in vectors['spam']])
    for index in range(n):
        all_prob['spam'][index] = ((1 + sum(vectors['spam'][:, index])) / (f_spam + total_spam_term))
    return all_prob


def evaluation(vectorized_test_dict, vectorized_train_dict, similarity_func=0, naive=False):
    """

    :param vectorized_test_dict: vectorize_data output for test data
    :param vectorized_train_dict: vectorize_data output for train data
    :param similarity_func: 0 if we use cosine similarity and 1 if we use tf-idf score
    :param naive: False if we use KNN and TRue if we use naive bayes
    :return:

    get predicted classes using models and call evaluation_metrics_calculator to show the evaluation metrics
    """
    # index classes
    indexed_classes = {c: index for index, c in enumerate(vectorized_test_dict.keys())}

    predicted_ls = list()
    true_ls = list()
    if not naive:
        if similarity_func == 0:
            # use knn to assign a label to each email an store it in predicted_ls list
            for vec in vectorized_test_dict['ham']:
                predicted_ls.append(knn(vec, similarity_func, vectorized_train_dict=vectorized_train_dict))
                true_ls.append('ham')
            for vec in vectorized_test_dict['spam']:
                predicted_ls.append(knn(vec, similarity_func, vectorized_train_dict=vectorized_train_dict))
                true_ls.append('spam')
        elif similarity_func == 1:
            idf = idf_calculator(vectorized_train_dict)
            tf_dict = tf_calculator(vectorized_train_dict)

            # use knn to assign a label to each email an store it in predicted_ls list
            for vec in vectorized_test_dict['ham']:
                predicted_ls.append(knn(vec, similarity_func, tf_dict=tf_dict, idf=idf))
                true_ls.append('ham')

            for vec in vectorized_test_dict['spam']:
                predicted_ls.append(knn(vec, similarity_func, tf_dict=tf_dict, idf=idf))
                true_ls.append('spam')
    else:
        all_prob_dict = all_probabilities_calculator(vectorized_train_dict)
        for vec in vectorized_test_dict['ham']:
            non_zer_indices = np.nonzero(vec)[0]
            prob_ham = 0
            prob_spam = 0
            # for each vec in test, calculate lg(p('ham'|vec)) and lg(p('spam'|vec)) using all_prob_dict
            # then compare them and choose max one as assigned class
            for index in non_zer_indices:
                prob_ham += np.log10(all_prob_dict['ham'][index] + 1)

            for index in non_zer_indices:
                prob_spam += np.log10(all_prob_dict['spam'][index] + 1)

            if prob_ham > prob_spam:
                predicted_ls.append('ham')
            else:
                predicted_ls.append('spam')
            true_ls.append('ham')
        # repeat for spam data in test docs
        for vec in vectorized_test_dict['spam']:
            non_zer_indices = np.nonzero(vec)[0]
            prob_ham = 0
            prob_spam = 0
            for index in non_zer_indices:
                prob_ham += np.log10(all_prob_dict['ham'][index] + 1)

            for index in non_zer_indices:
                prob_spam += np.log10(all_prob_dict['spam'][index] + 1)

            if prob_ham > prob_spam:
                predicted_ls.append('ham')
            else:
                predicted_ls.append('spam')
            true_ls.append('spam')

    evaluation_metrics_calculator(predicted_ls, true_ls, indexed_classes)


def evaluation_metrics_calculator(predicted_ls, true_ls, indexed_classes):
    # construct confusion matrix or ech class based on the order they've been indexed
    # its general and will work for more than three classes
    confusion_matrix = np.zeros((len(indexed_classes), 2, 2), dtype=int)
    for i in range(len(true_ls)):
        if true_ls[i] == predicted_ls[i]:
            # True Positive
            confusion_matrix[indexed_classes[true_ls[i]]][1][1] += 1
            # True Negative for other 2
            for c in indexed_classes.keys():
                if c != true_ls[i]:
                    confusion_matrix[indexed_classes[c]][0][0] += 1
        else:
            # False Positive:
            confusion_matrix[indexed_classes[true_ls[i]]][0][1] += 1
            # False Negative
            confusion_matrix[indexed_classes[predicted_ls[i]]][1][0] += 1
            # True Negative for the other class
            for c in indexed_classes.keys():
                if c != true_ls[i] and c != predicted_ls[i]:
                    confusion_matrix[indexed_classes[c]][0][0] += 1
    """
    TN  FP
    FN  TP
    macro: avg
    micro: calculate sum
    """

    print('class indexes: {}'.format(indexed_classes))
    print('confusion_matrix:    \n{}'.format(confusion_matrix))

    # calculate precision for each class using confusion matrix
    precision_per_class = dict()
    for c in indexed_classes.keys():
        if (confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][0][1]) != 0:
            p = float(confusion_matrix[indexed_classes[c]][1][1]) / float(
                confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][0][1])
        else:
            p = 0
        precision_per_class.update({c: p})

    # calculate recall for each class using confusion matrix
    recall_per_class = dict()
    for c in indexed_classes.keys():
        if (confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][1][0]) != 0:
            r = float(confusion_matrix[indexed_classes[c]][1][1]) / float(
                confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][1][0])
        else:
            r = 0
        recall_per_class.update({c: r})

    # calculate f1 for each class using confusion matrix
    f1_per_class = dict()
    for c in indexed_classes.keys():
        if (2 * confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][1][0] +
            confusion_matrix[indexed_classes[c]][0][1]) != 0:
            f1 = 2 * float(confusion_matrix[indexed_classes[c]][1][1]) / float(
                2 * confusion_matrix[indexed_classes[c]][1][1] + confusion_matrix[indexed_classes[c]][1][0] +
                confusion_matrix[indexed_classes[c]][0][1])
        else:
            f1 = 0
        f1_per_class.update({c: f1})

    # calculate error rate for each class using confusion matrix
    error_rate_per_class = dict()
    for c in indexed_classes.keys():
        error_rate = float(confusion_matrix[indexed_classes[c]][0][1] + confusion_matrix[indexed_classes[c]][1][0]) / \
                     float(sum(sum(confusion_matrix[indexed_classes[c]])))
        error_rate_per_class.update({c: error_rate})

    total = [[0, 0], [0, 0]]
    # calculate total TP, TN, FP and FN
    for c in indexed_classes.keys():
        total[0][0] += confusion_matrix[indexed_classes[c]][0][0]
        total[0][1] += confusion_matrix[indexed_classes[c]][0][1]
        total[1][0] += confusion_matrix[indexed_classes[c]][1][0]
        total[1][1] += confusion_matrix[indexed_classes[c]][1][1]

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
    f1_macro = float(sum(f1_per_class.values())) / float(len(indexed_classes))

    print('precision per class:     {}'.format(precision_per_class))
    print('recall per class:        {}'.format(recall_per_class))
    print('f1 per class             {}'.format(f1_per_class))
    print('error rate per class     {}\n'.format(error_rate_per_class))

    print('accuracy:                {}'.format(accuracy))
    print('f1 macro:                {}'.format(f1_macro))
    print('f1 micro:                {}'.format(f1_micro))


if __name__ == '__main__':
    print('model')
