"""
Обработка естественного языка
"""

import sys
import csv
import codecs
import re

vocabulary = set()

reviews = []
ratings = []

class PureText:

    def __init__(self, text):
        self.text = text

    def remove_numbers(self):
        self.text = re.sub('[0-9]+', '', self.text)
        return self

    def remove_html_tags(self):
        self.text = re.sub(r'<[^>]+>', '', self.text)
        return self

    def remove_symbols(self):
        self.text = re.sub("[-\?;,!@#$%^&*(){}£\/'']",'', self.text).replace('"', '').replace('...', ' ')\
            .replace('.', ' ').replace('..', ' ').replace('--', ' ')
        self.text = re.sub("( \?\(!@#$%^&*()_+=-'\:;|/`~.,{})",'', self.text)
        return self

with codecs.open("IMDB Dataset.csv", "r", "utf_8_sig" ) as file:
    for i, line in enumerate(file):
        line = line.split('",')

        if not line or len(line) < 2:
            continue

        review = PureText(line[0].strip()).remove_numbers().remove_html_tags().remove_symbols().text.strip()
        reviews.append(review)
        ratings.append(line[1].strip())

        words = set(review.split())
        words = set(word.lower() for word in words)
        vocabulary.update(words)

        if i == 24999:
            break

indexed_vocabulary = {}
for i, word in enumerate(vocabulary):
    indexed_vocabulary[word] = i

input_data_set = []
for review in reviews:
    review_indexes = []
    for word in review.split():
        review_indexes.append(indexed_vocabulary.get(word.lower()))

    input_data_set.append(review_indexes)


target_data_set = []
for rating in ratings:
    if rating == 'positive':
        target_data_set.append(1)
    else:
        target_data_set.append(0)

x = 4


