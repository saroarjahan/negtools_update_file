import numpy as np
import sklearn.metrics as metrics
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from nltk.corpus import words

from utilss import *

def extract_features_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon, mode='training'):
    """
    Extracts features for the cue classifier from the sentence dictionaries.
    Returns (modified) sentence dictionaries, a list of feature dictionaries, and
    if called in training mode, a list of labels. 
    """
    instances = []
    for sent in sentence_dicts:
        # print(sent)
        for key, value in sent.items():
            features = {}
            if isinstance(key, int):
                if not_known_cue_word(value[3].lower(), cue_lexicon, affixal_cue_lexicon):
                    sent[key]['not-pred-cue'] = True
                    continue

                features['token'] = value[3].lower()
                features['lemma'] = value[4].lower()
                features['pos'] = value[5]

                if key == 0:
                    features['bw-bigram1'] = 'null'
                else:
                    features['bw-bigram1'] = "%s_*" %sent[key-1][4].lower()
                if not (key+1) in sent:
                    features['fw-bigram1'] = 'null'
                else:
                    features['fw-bigram1'] = "*_%s" %sent[key+1][4].lower()
                    
                affix = get_affix_cue(value[3].lower(), affixal_cue_lexicon)
                if affix != None:
                    base = value[3].lower().replace(affix, "")
                    features['char-5gram1'], features['char-5gram2'] = get_character_ngrams(base, affix, 5)
                    features['char-4gram1'], features['char-4gram2'] = get_character_ngrams(base, affix, 4)
                    features['char-3gram1'], features['char-3gram2'] = get_character_ngrams(base, affix, 3)
                    features['char-2gram1'], features['char-2gram2'] = get_character_ngrams(base, affix, 2)
                    features['char-1gram1'], features['char-1gram2'] = get_character_ngrams(base, affix, 1)
                    features['affix'] = affix
                else:
                    features['char-5gram1'], features['char-5gram2'] = 'null','null'
                    features['char-4gram1'], features['char-4gram2'] = 'null','null'
                    features['char-3gram1'], features['char-3gram2'] = 'null','null'
                    features['char-2gram1'], features['char-2gram2'] = 'null','null'
                    features['char-1gram1'], features['char-1gram2'] = 'null','null'
                    features['affix'] = 'null'
                    
                instances.append(features)
    if mode == 'training':
        labels = extract_labels_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon)
        return sentence_dicts, instances, labels
    return sentence_dicts, instances


def extract_labels_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon):
    """
    Extracts labels for training the cue classifier. Skips the words that are not
    known cue words. For known cue words, label 1 means cue and label -1 means non
    cue. Returns a list of integer labels. 
    """
    labels = []
    for sent in sentence_dicts:
        for key, value in sent.items():
            if isinstance(key, int):
                if not_known_cue_word(value[3].lower(), cue_lexicon, affixal_cue_lexicon):
                    continue
                if any(cue_position == key for (cue, cue_position, cue_type) in sent['cues']) or any(mw_pos == key for (mw_cue, mw_pos) in sent['mw_cues']):
                    labels.append(1)
                else:
                    labels.append(-1)
    return labels
                
def extract_features_scope(sentence_dicts, mode='training'):
    """
    Extracts features for the scope classifier from the sentence dictionaries.
    Returns (modified) sentence dictionaries, a list of feature dictionaries,
    a list of the sentence lengths, and if called in training mode, a list of labels.
    """
    instances = []
    sentence_splits = []
    for sent in sentence_dicts:
        if not sent['neg']:
            continue
        print(sent)
        graph = make_dir_graph_for_sentence(sent)
        bidir_graph = make_bidir_graph_for_sentence(sent)
        for cue_i, (cue, cue_position, cue_type) in enumerate(sent['cues']):
            seq_length = -1
            for key, value in sent.items():
                features = {}
                if isinstance(key, int):
                    features['token'] = value[3]
                    features['lemma'] = value[4]
                    features['pos'] = value[5]
                    features['dir-dep-dist'] = get_shortest_path(graph, sent, cue_position, key)
                    features['dep-graph-path'] = get_dep_graph_path(bidir_graph, sent, cue_position, key)

                    dist = key - cue_position
                    nor_index = find_nor_index(sent)
                    if cue == "neither" and nor_index > -1 and abs(key-nor_index) < abs(dist):
                        dist = key - nor_index
                    #token is to the left of cue
                    if dist < 0:
                        if abs(dist) <= 9:
                            features['left-cue-dist'] = 'A'
                        else:
                            features['left-cue-dist'] = 'B'
                        features['right-cue-dist'] = 'null'
                    #token is to the right of cue
                    elif dist > 0:
                        if dist <= 15:
                            features['right-cue-dist'] = 'A'
                        else:
                            features['right-cue-dist'] = 'B'
                        features['left-cue-dist'] = 'null'
                    else:
                        features['left-cue-dist'] = '0'
                        features['right-cue-dist'] = '0'
                    features['cue-type'] = cue_type
                    features['cue-pos'] = sent[cue_position][5]

                    if key == 0:
                        features['bw-bigram1'] = 'null'
                        features['bw-bigram2'] = 'null'
                    else:
                        features['bw-bigram1'] = "%s_*" %sent[key-1][4]
                        features['bw-bigram2'] = "%s_*" %sent[key-1][5]
                    if not (key+1) in sent:
                        features['fw-bigram1'] = 'null'
                        features['fw-bigram2'] = 'null'
                    else:
                        features['fw-bigram1'] = "*_%s" %sent[key+1][4]
                        features['fw-bigram2'] = "*_%s" %sent[key+1][5]
                    instances.append(features)
                    if key > seq_length:
                        seq_length = key
            sentence_splits.append(seq_length)
    if mode == 'training':
        labels = extract_labels_scope(sentence_dicts, mode)
        return sentence_dicts, instances, labels, sentence_splits
    return sentence_dicts, instances, sentence_splits

def extract_labels_scope(sentence_dicts, config):
    """
    Extracts labels for training the scope classifier. Skips the sentences that
    do not contain a cue. Label values:
    In-scope: 0
    Out of scope: 1
    Beginning of scope: 2
    Cue: 3
    Returns a list of labels.
    """
    labels = []
    for sent in sentence_dicts:
        if not sent['neg']:
            continue
        for cue_i, (cue, cue_position, cue_type) in enumerate(sent['cues']):
            prev_label = 1
            for key, value in sent.items():
                if isinstance(key, int):
                    scope = sent['scopes'][cue_i]
                    if any(key in s for s in scope):
                        if prev_label == 1:
                            labels.append(2)
                            prev_label = 2
                        else:
                            labels.append(0)
                            prev_label = 0
                    elif key == cue_position:
                        labels.append(3)
                        prev_label = 3
                    else:
                        labels.append(1)
                        prev_label = 1
    return labels

