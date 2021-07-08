
import itertools
import argparse
import pickle
import numpy as np
from sklearn.externals import joblib

from utilss import get_affix_cue, count_multiword_cues, mwc_start, convert_to_IO, make_complete_labelarray, make_splits
from negtool import load_cue_learner, load_scope_learner
from file_reading import read_parsed_data, read_cuepredicted_data
from read_CD_file import read_CD_file
from feature_extraction import extract_features_scope, extract_features_cue 

def convert_cues_to_fileformat(sentences, labels, affix_cue_lexicon, filename, fileformat):
    infile = open(filename, "r")
    filename_base = filename.split("/")[-1].split(".")[0]
    output_filename = "%s_evaluation_cues.txt" %filename_base
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    upper_limit = 7
    n_cues = sum(i > 0 for i in labels[sent_counter])
    n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
    if has_mwc:
        n_cues += n_mwc - 1
    written_cues = n_cues*[False]
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            line_counter = 0
            if sent_counter < len(labels):
                n_cues = sum(i > 0 for i in labels[sent_counter])
                n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
                if has_mwc:
                    n_cues += n_mwc - 1
                written_cues = n_cues*[False]
            outfile.write("\n")
        else:
            if not fileformat == "CD":
                #delete unneccesary columns to match CD
                del tokens[3]
                del tokens[5]
                del tokens[6]
                tokens[0] -= str(int(tokes[0]) - 1) #start counting from 0
                tokens = ['_', '_'] + tokens #add to columns to the start to match CD
            written_cue_on_line = False
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            if n_cues == 0:
                outfile.write("***\n")
            else:
                for cue_i in range(n_cues):
                    if labels[sent_counter][line_counter] < 0:
                        outfile.write("_\t_\t_\t")
                    else:
                        if written_cues[cue_i] or written_cue_on_line:
                            outfile.write("_\t_\t_\t")
                        else:
                            affix = get_affix_cue(tokens[3].lower(), affix_cue_lexicon)
                            if affix != None:
                                outfile.write("%s\t_\t_\t" %affix)
                                written_cues[cue_i] = True
                            else:
                                outfile.write("%s\t_\t_\t" %tokens[3])
                                prev_token = sentences[sent_counter][line_counter-1][3].lower() if line_counter > 0 else 'null'
                                if not mwc_start(tokens[3].lower(), prev_token):
                                    written_cues[cue_i] = True
                            written_cue_on_line = True
                line_counter += 1
                outfile.write("\n")
    infile.close()
    outfile.close()

def convert_scopes_to_fileformat(filename, sentences, labels, fileformat):
    base_name = filename.split("/")[-1].split(".")[0]
    infile = open(filename, "r")
    outfile = open("%s_evaluation_scopes.txt" %base_name, "w")
    sent_counter = 0
    line_counter = 0
    scope_counter = 0
    upper_limit = 7
    n_cues = 0
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            scope_counter += n_cues
            line_counter = 0
            n_cues = 0
            outfile.write("\n")
        elif tokens[-1] == "***":
            outfile.write(line)
        else:
            sent = sentences[sent_counter]
            cues = sent['cues']
            n_cues = len(cues)
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[upper_limit + 3*cue_i])
                if labels[scope_counter][line_counter] == 0 or labels[scope_counter][line_counter] == 2:
                    if cues[cue_i][2] == 'a' and sent[int(cues[cue_i][1])][3] == tokens[3]:
                        outfile.write("%s\t" %(tokens[3].replace(cues[cue_i][0], "")))
                    elif tokens[upper_limit + 3*cue_i] != "_":
                        outfile.write("_\t")
                    else:
                        outfile.write("%s\t" %tokens[3])
                else:
                    outfile.write("_\t")

                outfile.write("%s\t" %tokens[upper_limit + 2 + 3*cue_i])
                scope_counter += 1
            
            scope_counter -= n_cues
            line_counter += 1
            outfile.write("\n")

    infile.close()
    outfile.close()

def load_cue_learner(cue_model, cue_vectorizer, cue_lexicon, affixal_cue_lexicon):
    cue_ssvm = pickle.load(open(cue_model, "rb"))
    cue_vect = joblib.load(cue_vectorizer)
    cue_lex = pickle.load(open(cue_lexicon, "rb"))
    affixal_cue_lex = pickle.load(open(affixal_cue_lexicon, "rb"))
    return cue_ssvm, cue_vect, cue_lex, affixal_cue_lex

def load_scope_learner(scope_model, scope_vectorizer):
    scope_ssvm = pickle.load(open(scope_model, "rb"))
    scope_vect = joblib.load(scope_vectorizer)
    return scope_ssvm, scope_vect

def test_cue_model(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, mode, fileformat, parsed_cd_file=None):
    """
    Reads the file with the input data, extracts features for cue detection,
    does cue prediction and converts the predicted cues to the CD file format
    """
    if fileformat == "CD":
        dev_sentence_dicts = read_CD_file(filename, parsed_cd_file)
    else:
        dev_sentence_dicts = read_parsed_data(filename, 'parsed')
    dev_sd, dev_instances = extract_features_cue(dev_sentence_dicts, cue_lexicon, affixal_cue_lexicon, 'prediction')
    dev_fvs = cue_vectorizer.transform(dev_instances).toarray()
    y_pred = cue_ssvm.predict(dev_fvs)
    dev_y_pred = make_complete_labelarray(dev_sentence_dicts, y_pred)
    convert_cues_to_fileformat(dev_sentence_dicts, dev_y_pred, affixal_cue_lexicon, filename, fileformat)

def test_scope_model(scope_ssvm, scope_vectorizer, filename, fileformat, parsed_cd_file=None):
    """
    Reads the file with the predicted cues, extracts features for scope resolution,
    does scope prediction and converts the predicted scopes for the predicted cues
    to the CD file format
    """
    if fileformat == "CD":
        sentence_dicts = read_CD_file(filename, parsed_cd_file)
    else:
        sentence_dicts = read_cuepredicted_data(filename, 'parsed')
    sentences, dev_instances, dev_splits = extract_features_scope(sentence_dicts, 'prediction')
    dev_fvs = scope_vectorizer.transform(dev_instances).toarray()
    X_dev, y_dev = make_splits(dev_fvs, [], dev_splits)
    y_pred = scope_ssvm.predict(X_dev)
    convert_scopes_to_fileformat(filename, sentences, y_pred, fileformat)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-cm', '--cuemodel', help="path to cue model object file",type=str)
    argparser.add_argument('-sm', '--scopemodel', help="path to scope model object file", type=str)
    argparser.add_argument('-cl', '--cuelex', help="path to cue lexicon object file", type=str)
    argparser.add_argument('-acl', '--affixalcuelex', help="path to affixal cue lexicon object file", type=str)
    argparser.add_argument('-cv', '--cuevect', help="path to cue vectorizer object file", type=str)
    argparser.add_argument('-sv', '--scopevect', help="path to scope vectorizer object file", type=str)
    argparser.add_argument('-tf', '--testfile', help="path to test file", type=str)
    argparser.add_argument('-cdf', '--cdfile', help="path to parsed CD file. Needs to be provided when evaluating CD files", nargs='?', type=str)
    argparser.add_argument('-ff', '--fileformat', help="fileformat of testfile. either CD or parsed", type=str)
    argparser.add_argument('-e2e', '--endtoend', help="end to end evaluation. If true, cues will be predicted and the scopes will be predicted for the predicted cues. If false, scopes will be predicted for gold cues.", type=bool)
    args = argparser.parse_args()

    filename_base = args.testfile.split("/")[-1].split(".")[0]

    cue_model, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = load_cue_learner(args.cuemodel, args.cuevect, args.cuelex, args.affixalcuelex)
    scope_model, scope_vectorizer = load_scope_learner(args.scopemodel, args.scopevect)
    if args.endtoend == 'true':
        test_cue_model(cue_model, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, args.testfile, 'parsed', args.fileformat, args.cdfile)
        test_scope_model(scope_model, scope_vectorizer, "%s_evaluation_cues.txt" %filename_base, args.fileformat, args.cdfile)
    else:
        test_scope_model(scope_model, scope_vectorizer, args.testfile, args.fileformat, args.cdfile)

