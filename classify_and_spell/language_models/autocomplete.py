import pickle
import math
import numpy as np
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, big_train_text, vocab, n=2, alpha=3e-3):
        # big train text = list of sentences split into individual words. 
        
        # get counts and perform any other setup
        self.n = n
        self.smoothing = alpha
        self.alpha = alpha
        self.total_vocab_len = len(vocab)
        self.vocab = vocab
    
        # Step 1. Get n-grams, and get the 
        self.counts= {}
#         for _ in range(n-1): 
#             train_text.insert(0, '<sos>')
        for train_text in big_train_text: 
            train_text = ['<s>']*(n-1) + train_text
            for k in range(n-1, len(train_text)): 
                prefix = (' ').join(train_text[(k-(n-1)):k])
                if not '<oov>' in prefix and not '<oov>' in train_text[k]:
                    if not prefix in self.counts: 
                        self.counts[prefix] = []
                    self.counts[prefix].append(train_text[k])
                
        self.totals = {}
        for k, v in self.counts.items(): 
            self.totals[k] = len(v)
            self.counts[k] = dict(Counter(v))
        

    def n_gram_probability(self, n_gram):
        """Return the probability of the last word in an n-gram.
        
        n_gram -- a list of string tokens
        returns the conditional probability of the last token given the rest.
        """

        if not len(n_gram) == self.n: 
            print(n_gram, len(n_gram), self.n)
        assert len(n_gram) == self.n
        prefix = (' ').join(n_gram[:-1])
        last_token = n_gram[-1]
        count = self.counts.get(prefix, 0)
        if not count == 0:
            num = (count.get(last_token, 0) + self.alpha)
        else: 
            num = self.alpha
        denom = (self.totals.get(prefix, 0) + (self.total_vocab_len)*self.alpha)
        res = num/denom
        return res
        
        

    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""
        # YOUR CODE HERE
        # use your function n_gram_probability
        # vocab.itos contains a list of words to return probabilities for
    
        if self.n == 1:
            prefix = []
        else: 
            text_prefix = ['<s>']*(self.n-1) + text_prefix.split(' ')
            prefix = text_prefix[-(self.n-1):]
        res = [self.n_gram_probability(prefix + [word]) for word in self.vocab]
        return res
        

    def perplexity(self, full_text):
        """ full_text is a list of string tokens
        return perplexity as a float """

        log_probabilities = []
        n = self.n
        big_train_text = full_text
        for train_text in big_train_text: 
            train_text = ['<s>']*(n-1) + train_text
            for k in range(n-1, len(train_text)): 
                prefix = train_text[(k-(n-1)):k]
                into_ngram = prefix + [train_text[k]]
#                 print('into ngram', into_ngram)
                log_probabilities.append(math.log(self.n_gram_probability(into_ngram), 2))
        return 2 ** -np.mean(log_probabilities)



class DiscountBackoffModel(NGramModel):
    def __init__(self, big_train_text, lower_order_model, vocab, n=2, delta=0.9):
        super().__init__(big_train_text, vocab=vocab, n=n)
        self.lower_order_model = lower_order_model
        self.discount = delta
                # Step 1. Get n-grams, and get the 
        self.counts= {}
#         for _ in range(n-1): 
#             train_text.insert(0, '<sos>')
        
        for train_text in big_train_text: 
            train_text = ['<s>']*(n-1) + train_text
            for k in range(n-1, len(train_text)): 
                prefix = (' ').join(train_text[(k-(n-1)):k])
                if not '<oov>' in prefix and not '<oov>' in train_text[k]:
                    if not prefix in self.counts: 
                        self.counts[prefix] = []
                    self.counts[prefix].append(train_text[k])
            
            
        self.totals = {}
        for k, v in self.counts.items(): 
            self.totals[k] = len(v)
            self.counts[k] = dict(Counter(v))

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        
        prefix = (' ').join(n_gram[:-1])
        last_token = n_gram[-1]
      
        count = self.counts.get(prefix, 0)
        if not count == 0:
            count = (count.get(last_token, 0))
            
        c_bottom = self.totals.get(prefix, 0)
        if c_bottom == 0:
            return self.lower_order_model.n_gram_probability(n_gram[1:])

        backoff_init= max(count-self.discount, 0)/(c_bottom)
        
        n_plus_one = len(self.counts.get(prefix))
        alpha  = (self.discount*n_plus_one)/c_bottom # calculate alpha
        prob_n_plus_2 = self.lower_order_model.n_gram_probability(n_gram[1:])# calc p n plus 2
        
        result = backoff_init + alpha * prob_n_plus_2
        return result

class KneserNeyBaseModel(NGramModel):
    def __init__(self, big_train_text, vocab):
        super().__init__(big_train_text, vocab, n=1)
    
        
        n_tilde = 2
        # Step 1. Get n-grams, and get the 
        self.counts= {}
        for train_text in big_train_text: 
            train_text = ['<s>']*(n_tilde-1) + train_text
            for k in range(n_tilde-1, len(train_text)): 
                prefix = (' ').join(train_text[(k-(n_tilde-1)):k])
                if not '<oov>' in prefix and not '<oov>' in train_text[k]:
                    if not train_text[k] in self.counts: 
                        self.counts[train_text[k]] = []
                    self.counts[train_text[k]].append(prefix)
            
            
        
        self.totals = {}
        self.tot_norm = 0
        for k, v in self.counts.items(): 
            
            
            self.counts[k] = dict(Counter(v))
            self.totals[k] = len(self.counts[k])
            self.tot_norm += self.totals[k]
     
        self.total_vocab_len = len(vocab)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 1
        
        p = (self.totals.get((' ').join(n_gram), 0) + self.alpha)/(self.tot_norm + self.total_vocab_len*self.alpha)
        if p > 1: 
            print(p, n_gram, self.totals.get((' ').join(n_gram)), len(self.counts))
        return p
    
    
def load_autocomplete(path=None):
    """
    Input: 
        path: the path where a saved DiscountBackoffModel N-Gram model, which uses a KneserNeyBase model for the unigram probabilites, 
        is stored. 
    
    Output: 
        A DiscountBackoffModel that gives next_word_probabilites for every word in the vocab, given a prefix. 
    """
    
    with open(path, 'rb') as f:
        lm = pickle.load(f)
        return lm

def save_autocomplete(model, path):
    """
    Saves the autocomplete language model.

    Parameters
    ----------
    model : object
        The language model object.
    path : str
        The path to save the file at.
    """

    with open(path, 'wb') as f:
        pickle.dump(model, f)
