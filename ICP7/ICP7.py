import requests as r
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import ne_chunk
from nltk.util import ngrams
from bs4 import BeautifulSoup

# ICP7 Part 1
url = 'https://en.wikipedia.org/wiki/Google'
html = r.get(url).content
unicode_str = html.decode("utf8")
encoded_str = unicode_str.encode("ascii",'ignore')
soup = BeautifulSoup(encoded_str, "html.parser")
text = soup.get_text()
text = text[text.find('});});') + 6:]

# ICP7 Part 2
with open("input.txt", "w") as file:
    file.write(text.strip())

# ICP7 Part 3
# Tokenization
with open("input.txt", "r") as file:
    tokenized_words = []
    raw_text = file.read()
    sentences = sent_tokenize(raw_text)
    for sentence in sentences:
        tokenized_words.extend(word_tokenize(sentence))
    print(f"Tokenized words: {tokenized_words}")

# POS
    print(f"POS Tags: {pos_tag(tokenized_words)}")

# Stemming
    ps = PorterStemmer()
    stemmed_words = []
    for word in tokenized_words:
        stemmed_words.append(ps.stem(word))
    print(f"Stemming: {stemmed_words}")

# Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in tokenized_words:
        lemmatized_words.append(lemmatizer.lemmatize(word))
    print(f"Lemmatization: {lemmatized_words}")

# Trigrams
    trigrams_generator = ngrams(tokenized_words, 3)
    trigrams = []
    for trigram in trigrams_generator:
        trigrams.append(trigram)
    print(f"Trigrams: {trigrams}")

# Named Entity Recognition
    named_entities = []
    # pos_tag(tokenized_words
    for pos in pos_tag(tokenized_words:
        try:
            named_entities.append(ne_chunk(pos))
        except IndexError:
            pass
    print(f"Named Entities: {named_entities}")

# ICP7 Part 4
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# Multinomial
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(f"Multinomial Score: {score}")

# 4a) KNN
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(f"KNN Score: {score}")

# 4b) bigram
tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# Multinomial
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(f"Multinomial (with bigram tfidf vectorizer) Score: {score}")

# 4c) stop words
tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

# Multinomial
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(f"Multinomial (no stop_words) Score: {score}")
