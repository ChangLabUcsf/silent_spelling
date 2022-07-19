# These are based on the UC Berkeley CS288 HW1 (Ngram language modelling)
# and the deeplearning.ai NLP course.

# imports
from collections import defaultdict, Counter
import numpy as np
import math
import tqdm
import random
import torch
from torch import nn
import torch.nn.functional as F
import pickle

from .autocomplete import * #save_autocomplete
import string
import math
import random
import numpy as np
import pandas as pd
import nltk
import nltk
import pickle
import re
nltk.download('punkt')
nltk.data.path.append('.')
import os

class UnigramModel:
    def __init__(self, train_text, vocab):
        self.counts = Counter(train_text)
        self.total_count = len(train_text)
#         print(self.total_count)
        self.vocab = vocab

    def probability(self, word):
        return self.counts[word] / self.total_count + 1e-15

    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""
        return [self.probability(word) for word in self.vocab]

    def perplexity(self, full_text):
        """Return the perplexity of the model on a text as a float.
        
        full_text -- a list of string tokens
        """
        log_probabilities = []
        for word in full_text:
            # Note that the base of the log doesn't matter 
            # as long as the log and exp use the same base.
            log_probabilities.append(math.log(self.probability(word), 2))
        return 2 ** -np.mean(log_probabilities)

    
def load_vocab(vocab_path=None): 
    """
    Input: a vocab path. 
    Returns: a list of the cleaned vocab (no punctuation)

    """

    with open(vocab_path,'r') as f: 
        v = f.readlines()
    v = [vv.split(' ')[0] for vv in v]
    v = [vv.replace('\n', '') for vv in v]
    vocab = list(set(v))
    vocab = [v.replace("'", "") for v in vocab]

    vocab = [re.sub(r'[^\w\s]','',s) for s in vocab]
    for let in ['i', 'a']: 
        if not let in vocab:
            vocab.append(let)
    vocab = sorted(vocab)
    return vocab

def get_ngram_lm(vocab_filepath, ngram_filepath, corpus_fp=None):
    """Todo define the new ngram corpus up here. """
    
    if vocab_filepath is None and ngram_filepath is None: 
        # Just return the LM we used for testing ...
        ...
        
    with open(corpus_fp + 'en_US.txt', "r") as f:
        data = f.read()

    with open(corpus_fp + 'movie_lines.txt', 'r', encoding='ISO-8859-1') as f: 
        lines = f.readlines()

    lines = [l.split('+')[-1] for l in lines]
    lines = [l[1:].lower() for l in lines]
    
    lines = [s.translate(str.maketrans('', '', string.punctuation)) for s in lines]
    bad_chars = [ '\x82',
     '\x85',
     '\x8a',
     '\x8c',
     '\x91',
     '\x92',
     '\x93',
     '\x94',
     '\x96',
     '\x97',
     '£',
     '¥',
     '«',
     '\xad',
     '²',
     '³',
     '·',
     '¹',
     'ß',
     'à',
     'á',
     'â',
     'ä',
     'ç',
     'è',
     'é',
     'ê',
     'í',
     'ï',
     'ñ',
     'ò',
     'ó',
     'ô',
     'õ',
     'ù',
     'ú',
     'û',
     'ü', 
    '\t'] + [str(k) for k in range(10)]

    # Cut lines where the line has words outside of our 1000 word dataset.

    word_vocab = set(load_vocab(vocab_filepath))

    lines_ = ''
    for l in lines:

        badflag = False
        for word in l[:-1].split(' '):
            if not word in word_vocab: 
                badflag = True
            for char in word: 
                if char in bad_chars: 
                    badflag = True
        if not badflag and len(l)>0: 
            lines_ += l 

    lines[:10]

#     print(len(lines_))
    lines_[:1000]

    
    def split_to_sentences(data):
        """
        Split data by linebreak "\n"



        Args:
            data: str

        Returns:
            A list of sentences
        """
      
        sentences = data.split('\n')



        # Additional clearning (This part is already implemented)
        # - Remove leading and trailing spaces from each sentence
        # - Drop sentences if they are empty strings.
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]

        return sentences    

    # test your code
    x = """
    I have a pen.\nI have an apple. \nAh\nApple pen.\n
    """

    split_to_sentences(x)

    # Tokenize the sentences. 
    def tokenize_sentences(sentences):
        """
        Tokenize sentences into tokens (words)

        Args:
            sentences: List of strings

        Returns:
            List of lists of tokens
        """

        # Initialize the list of lists of tokenized sentences
        tokenized_sentences = []
        ### START CODE HERE (Replace instances of 'None' with your code) ###

        # Go through each sentence
        for sentence in sentences:

            # Convert to lowercase letters
            sentence = sentence.lower()
            sentence = sentence.replace('.', '')

            # Convert into a list of words
            tokenized = nltk.word_tokenize(sentence)

            # append the list of words to the list of lists
            tokenized_sentences.append(tokenized)

        ### END CODE HERE ###

        return tokenized_sentences

    # test your code
    sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
    tokenize_sentences(sentences)


    def get_tokenized_data(data):
        """
        Make a list of tokenized sentences

        Args:
            data: String

        Returns:
            List of lists of tokens
        """

        sentences = split_to_sentences(data)

   
        tokenized_sentences = tokenize_sentences(sentences)

       

        return tokenized_sentences


    x = "Sky is blue.\nLeaves are green\nRoses are red."
    get_tokenized_data(x)

    # Load the CORNELL MOVIES dataset

    data[:1009]

    data += lines_



    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    # random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    len(tokenized_data)


    vocab = load_vocab(vocab_filepath)

    len(vocab)


    tokenized_data[-10:], tokenized_data[:10]

    ct = 0
    good_sents = []
    valid_sents = []
    valid_ct = 0
    for t in tokenized_data:
        cleaned_sent = []
        oflag = False
        for tt in t:
    #         print(tt)
            if not tt in set(vocab):
                cleaned_sent.append('<oov>')
                oflag = True
            else: 
                cleaned_sent.append(tt)
    #         print(cleaned_sent)

        if oflag or valid_ct >= 100:
            good_sents.append(cleaned_sent)
        if not oflag and valid_ct < 100:
            valid_sents.append(cleaned_sent)
            valid_ct +=1

    vset = set()
    for g in good_sents:
        for gg in g:
            vset.add(gg)



    good_sents[-5:]

    len(valid_sents)

    vs_for_eval = [(' ').join(v) for v in valid_sents]

#     with open('./twitter_evals.pkl', 'wb') as f: 
#         pickle.dump(vs_for_eval, f)

    train_text = []
    for t in good_sents: 
        train_text.extend([tt for tt in t if not tt == '<oov>'])

    validation_text = []
    for t in valid_sents: 
        validation_text.extend([tt for tt in t if not tt == '<oov>'])





    unigram_demonstration_model = UnigramModel(train_text, vocab)
#     print('unigram validation perplexity:', 
#           unigram_demonstration_model.perplexity(validation_text, vocab))

    def check_validity(model, vocab):
        """Performs several sanity checks on your model:
        1) That next_word_probabilities returns a valid distribution
        2) That perplexity matches a perplexity calculated from next_word_probabilities

        Although it is possible to calculate perplexity from next_word_probabilities, 
        it is still good to have a separate more efficient method that only computes 
        the probabilities of observed words.
        """

        log_probabilities = []
        for i in range(10):
            prefix = validation_text[:i]
            probs = model.next_word_probabilities(prefix)
            assert min(probs) >= 0, "Negative value in next_word_probabilities"
            assert max(probs) <= 1 + 1e-8, "Value larger than 1 in next_word_probabilities"
    #         print(abs(sum(probs)))
            assert abs(sum(probs)-1) < 1e-4, "next_word_probabilities do not sum to 1"

            word_id = vocab.index(validation_text[i])
            selected_prob = probs[word_id]
            log_probabilities.append(math.log(selected_prob))

        perplexity = math.exp(-np.mean(log_probabilities))
        your_perplexity = model.perplexity(validation_text[:10])
        assert abs(perplexity-your_perplexity) < 0.1, "your perplexity does not " + \
        "match the one we calculated from `next_word_probabilities`,\n" + \
        "at least one of `perplexity` or `next_word_probabilities` is incorrect.\n" + \
        f"we calcuated {perplexity} from `next_word_probabilities`,\n" + \
        f"but your perplexity function returned {your_perplexity} (on a small sample)."


    check_validity(unigram_demonstration_model, vocab)

    def generate_text(model, n=20, prefix=('', '')):
        prefix = list(prefix)
        for _ in range(n):
            probs = model.next_word_probabilities(prefix)
            word = random.choices(vocab, probs)[0]
            prefix.append(word)
        return ' '.join(prefix)

#     print(generate_text(unigram_demonstration_model))

    n = 3
    for train_text in good_sents[:10]: 
#         print('train text', train_text)
        train_text = ['<s>']*(n-1) + train_text
        for k in range(n-1, len(train_text)): 
            prefix = (' ').join(train_text[(k-(n-1)):k])
            if not '<oov>' in prefix and not '<oov>' in train_text[k]:
#                 print(prefix, '//', train_text[k])
                ...
                

    train_text = [g for g in good_sents if not g in valid_sents]
    validation_text = valid_sents[:]

    validation_text[:3]

    
    
    # Now we train up the NGram models on the text. These are defined in the autocomplete script. 
    
    unigram_model = NGramModel(train_text, vocab, 1)
    # check_validity(unigram_model)
#     print('unigram validation perplexity:', unigram_model.perplexity(validation_text)) 

    bigram_model = NGramModel(train_text, vocab, n=2)
#     print('bigram validation perplexity:', bigram_model.perplexity(validation_text))

    trigram_model = NGramModel(train_text, vocab, n=3)
    # check_validity(trigram_model)
#     print('trigram validation perplexity:', trigram_model.perplexity(validation_text)) # this won't do very well...


    import time
    s = time.time()
    v = trigram_model.next_word_probabilities('how are')
#     print(type(v))
    print('how are...lm prediction:', vocab[np.argmax(v)])
    e = time.time()
#     print('ngram prediction time', e-s)


    bigram_backoff_model = DiscountBackoffModel(train_text, unigram_model, vocab, 2)
    trigram_backoff_model = DiscountBackoffModel(train_text, bigram_backoff_model, vocab, 3)
    # quadgram_backoff_model = DiscountBackoffModel(train_text, trigram_backoff_model, vocab, 4)
#     print('trigram backoff validation perplexity:', trigram_backoff_model.perplexity(validation_text))

    s = time.time()
    for _ in range(1):
        v = trigram_backoff_model.next_word_probabilities('what do you')
        print('what do you...lm prediction:', vocab[np.argmax(v)])
    e= time.time()
#     print(e-s)

    # Now add the KneserNey base
    kn_base = KneserNeyBaseModel(train_text, vocab)
    # check_validity(kn_base)
    bigram_kn_backoff_model = DiscountBackoffModel(train_text, kn_base, vocab, 2)
    trigram_kn_backoff_model = DiscountBackoffModel(train_text, bigram_kn_backoff_model, vocab, 3)
#     print('trigram Kneser-Ney backoff validation perplexity:', trigram_kn_backoff_model.perplexity(validation_text))

    s = time.time()
    for _ in range(10):
        v = trigram_backoff_model.next_word_probabilities('hello')
#     print(vocab[np.argmax(v)])
    e= time.time()
#     print(e-s)


    print('Ngram LM trained...saving Ngram LM')
    save_autocomplete(trigram_kn_backoff_model, 
                     ngram_filepath)
