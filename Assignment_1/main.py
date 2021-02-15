import pickle

from constants import given_data_root_path
from classifier import advance_classifier_training, basic_classifier_training, evaluation, predict

if __name__ == '__main__':

    action = ''
    acceptable_input_code = {'ea', 'eb', 'pa', 'pb', 'q'}
    while action.lower() != 'q':
        action = input('SELECT ONE OPTION AMONG THESE\n'
                       + '\t- ea : see evaluation result for airline-test.csv data for advance model\n '
                       + '\t- pa : insert a tweet text and see the result  for advance model\n'
                       + '\t- eb : see evaluation result for airline-test.csv data for basic model\n '
                       + '\t- pb : insert a tweet text and see the result  for basic model\n'
                       + '\t- q : quit  \n')
        if not (action.lower() in acceptable_input_code):
            print(' Ops, you just entered sth wrong, lets try again \n')
            continue
        else:
            if action.lower() == 'ea':
                try:  # check if model has been trained or not
                    with open('svm_advance_model.pickle', 'rb') as svm_model_file:
                        pickle.load(svm_model_file)
                except IOError:
                    print('model not exists, train first')
                    advance_classifier_training(given_data_root_path + 'airline-train.csv',
                                                given_data_root_path + 'airline-dev.csv')

                evaluation(given_data_root_path + 'airline-test.csv', advance=True)
            elif action.lower() == 'pa':
                try:  # check if model has been trained or not
                    with open('svm_advance_model.pickle', 'rb') as svm_model_file:
                        pickle.load(svm_model_file)
                except IOError:
                    print('model not exists, train first')
                    advance_classifier_training(given_data_root_path + 'airline-train.csv',
                                                given_data_root_path + 'airline-dev.csv')
                tweet = input('insert tweet(text)\n')
                print('assigned class:      {}\n'.format(predict(tweet, advance=True)))

            elif action.lower() == 'eb':
                try:  # check if model has been trained or not
                    with open('svm_basic_model.pickle', 'rb') as svm_model_file:
                        pickle.load(svm_model_file)
                except IOError:
                    print('model not exists, train first')
                    basic_classifier_training(given_data_root_path + 'airline-train.csv',
                                              given_data_root_path + 'airline-dev.csv')

                evaluation(given_data_root_path + 'airline-test.csv', advance=False)
                print()
            elif action.lower() == 'pb':
                try:  # check if model has been trained or not
                    with open('svm_basic_model.pickle', 'rb') as svm_model_file:
                        pickle.load(svm_model_file)
                except IOError:
                    print('model not exists, train first')
                    basic_classifier_training(given_data_root_path + 'airline-train.csv',
                                              given_data_root_path + 'airline-dev.csv')
                tweet = input('insert tweet(text)\n')
                print('assigned class:      {}\n'.format(predict(tweet, advance=False)))
            print()
