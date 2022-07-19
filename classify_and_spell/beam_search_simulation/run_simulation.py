import time
import numpy as np
import re

class VALID(object):
    def __init__(self, vocab, dec_alphabet, scorer):
        """
        Spelling beam search
        """
        self.vocab = vocab
        self.dec_alphabet = dec_alphabet
        self.full_vocs = self.initialize_full_vocab(vocab)
        self.trimmed_vocs = self.initialize_trimmed_vocab(vocab)
        self.scorer = scorer

        # initialize the dicts that will be used throughout. 
    # Initialization functions.
    def decode(self, list_of_inds):
        return [self.dec_alphabet[ind] for ind in list_of_inds]
        """given a list of indices, return a list of characters.
        The characters still need to be joined after this..."""

    def initialize_full_vocab(self, vocab):
        """Possible full character paths"""
        full_vocs = {}
        for k in range(1, max([len(v) for v in vocab])+1):
            full_vocs[k] = set([v for v in vocab if len(v) == k])
        return full_vocs

    def initialize_trimmed_vocab(self, vocab):
        """Possible partial character paths"""
        trimmed_vocs = {}
        for k in range(1, max([len(v) for v in vocab])+1):
            trimmed_vocs[k] = set([(v[:k]) for v in vocab if len(v) >= k])
        trimmed_vocs[0] = ''
        return trimmed_vocs

    def check_valid(self, strang):
        """
        Check if an output sequence is valid, by checking its in the trimmed vocabulary
        with the corresponding length.
        """
        add_flag = True
        for w_i, word in enumerate(strang.split(' ')):
            if w_i == len(strang.split(' ')) -1: 
                # the word can be partially done
                if not word in self.trimmed_vocs.get(len(word), []):
                    add_flag= False
            else: 
                if not word in self.full_vocs.get(len(word), []):
                    add_flag = False


        return add_flag

    def check_full(self, strang):
        """
        Check a sequence of characters after all the prdictions have been made

        Only allow sequences corresponding to finalized words. 

        This should only be called at the end of the sentence, it basically just serves
        to clip that final wrd.
        """
        add_flag = True
        for ff in (strang.split(' ')):
            if len(ff) == 0:
                add_flag = False
            elif ff in self.full_vocs.get(len(ff), []):
                0
            else: 
                add_flag=False

        return add_flag


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

def lmscore(sent, lm, vocab):
    sent = sent.strip()
    prefix = ' '.join(sent.split(' ')[:-1])
    word = sent.split(' ')[-1]
    if len(prefix) ==0:
        prefix = '<s>'
    return (lm.next_word_probabilities(prefix)[vocab.index(word)])


chars= list('abcdefghijklmnopqrstuvwxyz')
chars = chars + [c+' ' for c in chars]
dec_alphabet= {
    k:v for k,v in enumerate(chars)
}    


def prefix_search(matrix, 
                  lm=None, 
                  k=None, 
                  alpha=None, 
                  beta=None, 
                  prune=0.001, 
                 final_lm_alpha=1.0, 
                 scorer=None,
                 validator=None,
                 vocab=None,
                 no_constraints=False,
                 block = None):

    """
    This simulates the beam search that we ran in realtime. 
    
    As input it takes: 
    
        matrix, the Timesteps by Codewords matrix of the probabilities. 
        k = the beam width
        alpha = LM weight during the search
        beta = Word insertion bonus
        prune = unu
    
    """
    # Step 1. Duplicate the matrix along the last axis, this then gives probability to the characters with the space
    # after them. 
    ctc = np.log(np.hstack((matrix[:, :26], matrix[:, :26])))
    T = ctc.shape[0]
    O = ''
    prefixes = [('', 0)]
    prev_best = ''
    out_of_prefixes = False
    out_of_prefix_ind = 0
    
    
    for t in range(T):

        s = time.time()
        # This was always the case
        # Go through predictions at time t above .001
        pruned_alphabet = [chars[i] for i in np.where(ctc[t] > -6.9)[0]]
        new_prefixes = []
        token_probs = ctc[t]
        
        # Adjust for how we handled low number of pruned characters on 1st day vs other days. 
        if block > 2734:  
            if len(pruned_alphabet) < 6: 
#                 print('expanding pruned alpha')
                pruned_alphabet = [chars[i] for i in np.argsort(token_probs)[::-1][:13]]
            else: 
                ...
                #print('no expansion, pruned_alphabet len', len(pruned_alphabet))
        
        
        # For all the prefixes, try adding it to the beam. Check thats valid. 
        
        for l, score in prefixes:
            for c in pruned_alphabet: 
                c_ix = chars.index(c)
                l_plus = l + c
                if no_constraints or validator.check_valid(l_plus):
                    if c.endswith(' ') and len(l.replace(' ', '')) > 0:
                        
                        # If theres a space, then score this with the LM
                        if not no_constraints:
                            ls = lmscore(l_plus, lm, vocab) # lmscore only returns odds of the last word given earlier. 
                        else: 
                            ls = 1 # dont add any new score. 
                        new_score = score+ alpha*np.log(ls) + ctc[t,c_ix]
                    else: 
                        new_score = score + (ctc[t, c_ix])

                    new_prefixes.append((l_plus, new_score))

        sorter = lambda v: v[1] + beta*np.log(len(v[0].strip().split(' ')) + 1) # Word insertion bonus. 
        new_prefixes = sorted(new_prefixes, key=sorter, reverse=True)[:k]
        
        # Handle the different way we adjusted for running out of prefixes on 1st vs other days. 
        if len (new_prefixes) == 0:
            if block > 2734:
                out_of_prefixes = True
                out_of_prefix_ind = t
            else: 
                pre = ''
                for c in ctc[:t+1]: 
                    pre += chars[np.argmax(c[:26])]
                new_prefixes = [(pre, -33)]
                
        # Output the most likley character at each step. 
        print('current best:', new_prefixes[0][0])
                
        if out_of_prefixes: 
            len_already = len(prev_best.replace(' ', ''))
            strang = ''
            for c in ctc[len_already:t+1]:
                strang += self.chars[np.argmax(c[:26])]
            new_prefixes = [(prev_best + chars[np.argmax(ctc[t])], -33)]
                
        
        if not out_of_prefixes:
            prev_best = new_prefixes[0][0]
        prefixes = new_prefixes
        e = time.time()
        
    if out_of_prefixes: 
        return new_prefixes[0]
    
    
    # Finalize the decoding. 
    if not no_constraints: 
        prefixes_ = [p for p in prefixes if validator.check_full(p[0])]  

    if len(prefixes_) == 0:
        print('no valid')
        prefixes_ = prefixes
        if len(prefixes_) == 0:
            return (prev_best, 0)
    else: 
        sents_to_score = []
        for txt, _ in prefixes_:   
            sents_to_score.append(txt)

        LM_scores = scorer.sentence_score(sents_to_score, log=True)
        new_hypotheses = [(sh[0], sh[1]+ final_lm_alpha*LM_scores[k]) for k, sh in enumerate(prefixes_)]
        output_sequences = sorted(new_hypotheses, key= lambda val:val[1], reverse=True)
        output_sequences = [(o[0], o[-1]) for o in output_sequences]
        prefixes_ = output_sequences

    return prefixes_[0]       

def set_config(ind, config, blocks, hardset_lms, hardset_beams):
    block = blocks[ind]
    config_two = bool(block > 2720)
    
    if config_two: 
        if not hardset_beams: 
            config['beam_width'] = 739
        if not hardset_lms: 
            config['alpha'] = 0.7442369
            config['beta'] = 4.028704898
            config['final_lm_alpha'] = 1.128856
        
    else: 
        if not hardset_beams:
            config['beam_width'] = 457
        if not hardset_lms:
            config['alpha'] = .64192
            config['beta'] = 10.524
            config['final_lm_alpha'] = 1.5268
    config['lm_candidates'] = config['beam_width']
    return config

    
def run_prefix_search(good_preds, config, lm,  blocks, vocab_filepath=None, no_constraints=False, useRTparams=False, hardset_lms=False, hardset_beams=False, greedy_flag=False):
    """
    Input: 
        good_preds: list of sents and a matrix of predictions for them.
        config: Language model parameters. 
        lm: the language model to score things with in the middle.
        vocab_filepath: 
        
    and a configuration dict for the LM hyperparameters
    Output: A list of lists (By block) of tuples of (ground_truth_sentence, decoded_sentence)
    """
    print('loading vocab', vocab_filepath)
    vocab = load_vocab(vocab_filepath)
    validator = VALID(vocab, dec_alphabet, None)
    
    from lm_scorer.models.auto import AutoLMScorer as LMScorer
    scorer = LMScorer.from_pretrained("distilgpt2", device='cpu', batch_size=128)
    gts, decodes = [], []
    for pred_ind, p in enumerate(good_preds):
        real = p[0]
        if useRTparams:
            config = set_config(pred_ind, config, blocks, hardset_lms, hardset_beams)
#         print('using config', config)
        
        if not greedy_flag: 
            dec = prefix_search(np.array(p[1]),
                                lm,
                                k=int(config['beam_width']),
                               alpha=config['alpha'], 
                               beta=config['beta'],
                               final_lm_alpha=config['final_lm_alpha'],
                               validator =validator,
                               scorer=scorer,
                                vocab=vocab,
                                no_constraints = no_constraints, 
                                block = blocks[pred_ind]
                               )
        else: 
            dec = ''
            for vec in p[1]:
                print(vec.shape)
                dec += chars[np.argmax(vec[:26])]
            dec = (dec, 100)
        gts.append(real)
        decodes.append(dec[0])
        print('Ground truth:', real)
        print('Final prediction with distil gpt2 applied:', dec[0])
        
    gts = [str(gt) for gt in gts]
    decodes = [str(d) for d in decodes]
    return gts, decodes
                    
    