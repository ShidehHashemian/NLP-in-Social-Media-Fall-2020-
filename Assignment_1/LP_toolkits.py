
from constants import word_re, emoticon_re, amp, useless_char
from nltk.corpus import stopwords

# Global Parameters
stop_words = set(stopwords.words('english'))


def tokenize(text):
    text = text.replace(amp, " and ")
    words = word_re.findall(text)

    # Possible alter the case, but avoid changing emoticons like :D into :d:
    # if not self.preserve_case:
    words = map((lambda x: x if emoticon_re.search(x) else x.lower()), words)

    words_new = list()
    for word in words:
        if word not in stop_words:
            if len(word) > 3:
                words_new.append(word)

    words = ' '.join(words_new)
    words = words.replace('¡', ' ! ')
    for char in useless_char:
        words = words.replace(char, '')
    words = words.split(' ')
    return words


if __name__ == '__main__':
    print('KPtoolkits')

