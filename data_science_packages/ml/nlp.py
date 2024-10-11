__author__ = 'Brett Allen (brettallen777@gmail.com)'

from nltk.corpus import stopwords
import os
import re
import pandas as  pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel 
import spacy
from spacy.cli.download import download
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Union, List

# Load spacy model
# NOTE: Experiment with different spacy model sizes:
#       "en", "en_core_web_sm", "en_core_web_md", and "en_core_web_lg"
spacy_model = "en_core_web_sm"

try:
  # Initialize spacy model, keeping only tagger component (for efficiency)
  nlp = spacy.load(spacy_model, disable=['parser', 'ner'])
except OSError:
  download(model=spacy_model)
  nlp = spacy.load(spacy_model, disable=['parser', 'ner'])

def sent_to_words(sentences: list):
  for sentence in sentences:
    # NOTE: deacc = True, removes punctuations
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts: list, stop_words: list=None):
    if not stop_words:
        stop_words = stopwords.words('english')
        
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_model):
    return [bigram_model[doc] for doc in texts]

def make_trigrams(texts, bigram_model, trigram_model):
    return [trigram_model[bigram_model[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# TODO Allow for N-Gram configuration (e.g., bigram or trigram)
def preprocess_data(data: list, stop_words: list=None):
    if not data:
        return None, None
    
    if not stop_words:
        stop_words = stopwords.words('english')

    # Remove new line characters 
    _data = [re.sub(r'\n', ' ', sent) for sent in data]

    # Remove distracting single quotes 
    _data = [re.sub(r"'", "", sent) for sent in _data]
    # pprint(_data[:1])

    # Tokenize and cleanup text
    data_words = list(sent_to_words(_data))
    # print(data_words[:1])

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    # print(trigram_mod[bigram_mod[data_words[0]]])

    # Remove Stop Words
    data_words_nostops = remove_stopwords(texts=data_words, stop_words=stop_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(texts=data_words_nostops, bigram_model=bigram_mod)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        texts=data_words_bigrams, 
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    )

    # print(data_lemmatized[:1])

    # Create Dictionary 
    id2word = corpora.Dictionary(data_lemmatized)  

    # Create Corpus 
    texts = data_lemmatized  

    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]
    return id2word, corpus

def create_wordcloud(data: Union[str, List, pd.Series], max_font_size: int=40, interpolation: str="bilinear"):
    if data is None or data == '':
      return
   
    text = data
    if not isinstance(text, str):
        if isinstance(text, pd.Series):
            text = text.values.tolist()

        text = " ".join(text).replace('\n', ' ')
    
    wordcloud = WordCloud(max_font_size=max_font_size).generate(text)

    # Display wordcloud
    plt.figure()
    plt.imshow(wordcloud, interpolation=interpolation)
    plt.axis("off")
    plt.show()

# TODO Add functions for topic modeling (building LDA model and visualization)
