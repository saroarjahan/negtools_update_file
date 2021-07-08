import sys
import subprocess

def run_corenlp(corenlp_path, filename):
    """
    Run corenlp for file with raw text.
    """
    with open(filename, 'r') as infile:
        line1 = infile.readline()
        line2 = infile.readline()
        if line1.split()[0] == '1' and line2.split()[0] == '2':
            print("ERROR: File seems to be parsed already")
            sys.exit(1)
            
    absolute_path = corenlp_path + "/*"
    args = ['java', '-cp', absolute_path, '-Xmx1800m', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',  '-annotators', 'tokenize,ssplit,pos,lemma,depparse', '-file', filename, '-outputFormat', 'conll']
    pipe = subprocess.call(args)

def read_parsed_data(filename, mode):
    """
    Read input file on CoNLL-X format and create sentence dictionaries. 
    If file is parsed with corenlp, there will be 7 columns, not 8 as for conll-x. 
    """
    with open(filename, 'r') as infile:
        sentence = {}
        counter = 0
        instances = []
        for line in infile:
            token_dict = {}
            tokens = line.split()
            if len(tokens) == 0:
                for key in sentence:
                    head_index = int(sentence[key]['head']) - 1
                    if head_index > -1:
                        sentence[key]['head-pos'] = sentence[head_index][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]

                instances.append(sentence)
                sentence = {}
                counter = 0
                continue
            if counter == 0 and tokens[0] != '1':
                print ("ERROR: Trying to read file that is not (properly) parsed")
                sys.exit(1)
            for i in range(3):
                if tokens[i] != "_":
                    token_dict[i+2] = tokens[i]
            if mode == 'raw':
                token_dict[5] = tokens[3]
                token_dict['head'] = tokens[5]
                token_dict['deprel'] = tokens[6]
            else:
                token_dict[5] = tokens[4]
                token_dict['head'] = tokens[6]
                token_dict['deprel'] = tokens[7]
            sentence[counter] = token_dict
            counter += 1
    return instances
            
def read_cuepredicted_data(filename, mode):
    """
    Read file with predicted cues and create sentence dictionaries for scopes
    """
    if mode == 'raw':
        lower_limit = 4 #end of columns with data that can be automatically copied
        upper_limit = 6 #end of original data: next index is start of cue-column
    else:
        lower_limit = 3 
        upper_limit = 7
    cue_offset = upper_limit - 5 #to simplify modulo calculations
    with open(filename, 'r') as infile:
        sentence = {}
        cues = []
        mw_cues = []
        line_counter = 0
        counter = 0
        cue_counter = 0
        prev_cue_column = -1
        instances = []

        for line in infile:
            token_dict = {}
            tokens = line.split()
            #check for sentence end
            if len(tokens) == 0:
                for key in sentence:
                    #store the head index and pos for each token in sentence
                    head_index = int(sentence[key]['head']) - 1
                    if head_index > -1:
                        sentence[key]['head-pos'] = sentence[head_index][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]
                sentence['cues'] = cues
                sentence['mw_cues'] = mw_cues
                if len(cues) > 0:
                    sentence['neg'] = True
                else:
                    sentence['neg'] = False

                instances.append(sentence)
                sentence = {}
                counter = 0
                prev_cue_column = -1
                cues = []
                mw_cues = []
                line_counter += 1
                continue

            for i in range(len(tokens)):            
                if tokens[i] != "_" and i < lower_limit:
                    #add an offset of 2 to make token dicts match the original CD dicts
                    token_dict[i+2] = tokens[i]
                #cue column
                elif tokens[i] != "***" and tokens[i] != "_" and i > upper_limit and (i-cue_offset) % 3 == 0:
                    if i == prev_cue_column:
                        #same column has another cue token. cue must be mw
                        cues[-1][2] = 'm'
                        prev_cue_column = i
                        mw_cues.append([cues[-1][0],cues[-1][1]])
                        mw_cues.append([tokens[i], counter])
                    elif tokens[i] != tokens[1]:
                        #cue does not match current token. must be affixal cue
                        cues.append([tokens[i], counter, 'a'])
                        prev_cue_column = i
                    else:
                        cues.append([tokens[i], counter, 's'])
                        prev_cue_column = i
            if mode == 'raw':
                token_dict['head'] = tokens[5]
                token_dict['deprel'] = tokens[6]
            else:
                token_dict[5] = tokens[4] #for conll-x data: record only the pos-tag, not cpos-tag
                token_dict['head'] = tokens[6]
                token_dict['deprel'] = tokens[7]

            sentence[counter] = token_dict
            counter += 1
            line_counter += 1
        return instances

