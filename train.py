import argparse
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from pystruct.models import ChainCRF, BinaryClf
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM
from pystruct.utils import SaveLogger

from file_reading import *
from feature_extraction import extract_features_scope, extract_features_cue 
from utilss import *
from file_writing import *
from read_labelled_data import read_file
from negtool import run_cue_learner, run_scope_learner

def train_cue_learner(sentence_dicts, C_value):
    cue_lexicon, affixal_cue_lexicon = get_cue_lexicon(sentence_dicts)
    cue_sentence_dicts, cue_instances, cue_labels = extract_features_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon, 'training')
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(cue_instances).toarray()
    model = BinaryClf()
    cue_ssvm = NSlackSSVM(model, C=C_value, batch_size=-1)
    cue_ssvm.fit(fvs, np.asarray(cue_labels))
    return cue_ssvm, vectorizer, cue_lexicon, affixal_cue_lexicon

def train_scope_learner(sentence_dicts, C_value):
    scope_sentence_dicts, scope_instances, scope_labels, sentence_splits = extract_features_scope(sentence_dicts, 'training')
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(scope_instances).toarray()
    X_train, y_train = make_splits(fvs, scope_labels, sentence_splits)
    model = ChainCRF()
    scope_ssvm = FrankWolfeSSVM(model=model, C=C_value, max_iter=10)
    scope_ssvm.fit(X_train, y_train)
    return scope_ssvm, vectorizer

def save_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename):
    pickle.dump(cue_ssvm, open("cue_model_%s.pkl" %filename, "wb"))
    joblib.dump(cue_vectorizer, "cue_vectorizer_%s.pkl" %filename)
    pickle.dump(cue_lexicon, open("cue_lexicon_%s.pkl" %filename, "wb"))
    pickle.dump(affixal_cue_lexicon, open("affixal_cue_lexicon_%s.pkl" %filename, "wb"))

def save_scope_learner(scope_ssvm, scope_vectorizer, filename):
    pickle.dump(scope_ssvm, open("scope_model_%s.pkl" %filename, "wb"))
    joblib.dump(scope_vectorizer, "scope_vectorizer_%s.pkl" %filename)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model', help="model to train. Either cue, scope or all", type=str, choices=['cue', 'scope', 'all'])
    argparser.add_argument('-tf', '--trainingfile', help="filename of training file", type=str)
    argparser.add_argument('-cp', '--cueparameter', help="regularisation parameter for the cue model", type=float, nargs="?", default=0.20)
    argparser.add_argument('-sp', '--scopeparameter', help="regularisation parameter for the scope model", type=float, nargs="?", default=0.20)
    args = argparser.parse_args()

    sentence_dicts = read_file(args.trainingfile)
    filename = args.trainingfile.split(".")[0]
    if args.model == 'cue' or args.model == 'all':
        cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = train_cue_learner(sentence_dicts, args.scopeparameter)
        save_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename)

    if args.model == 'scope' or args.model == 'all':
        scope_ssvm, scope_vectorizer = train_scope_learner(sentence_dicts, 0.20)
        save_scope_learner(scope_ssvm, scope_vectorizer, filename)

