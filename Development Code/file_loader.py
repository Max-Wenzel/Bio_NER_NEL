import nltk as nl
import os
from unidecode import unidecode
from glob import glob
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# def process_text(word):
#     word = word.lower()
#     return unidecode(ps.stem(word))

def process_text(word):
    return unidecode(word)


def create_file(path_name, temp_name, return_=False):
    file_data = []
    sentence_break = '_S_B_'

    # load all files in given directory into a list
    for file_name in glob(path_name + '/*.tsv'):
        file = []
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                par_line = line[:-1].split('\t')
                if par_line[0] != '-DOCSTART-':
                    if len(par_line) == 1:
                        file.append([sentence_break, 'O'])
                    else:
                        file.append([process_text(par_line[0]), par_line[3]])  # process each word
        file_data.append((file, file_name))
    data_pos = [[sentence_break, sentence_break, 'O']]

    # combine all files into a single list
    for f in file_data:
        sentence = []
        for word in f[0][1:]:
            if word[0] == sentence_break:
                pos = nl.pos_tag([w[0] for w in sentence])
                for ii in range(len(sentence)):
                    sentence[ii][1] = pos[ii][1]
                data_pos += (sentence + [[sentence_break, sentence_break, 'O']])
                sentence = []
            else:
                sentence.append([word[0], '', word[1]])

    # save as new file of all data together
    if os.path.exists(temp_name):  # replace files instead of adding on
        os.remove(temp_name)

    with open(temp_name, 'a') as f:
        for line in data_pos:
            if line[2][0] not in ['I', 'O', 'B']:
                f.write("{}\t{}\t{}\n".format(line[0], line[1], 'O'))
            else:
                f.write("{}\t{}\t{}\n".format(line[0], line[1], line[2]))
    if return_:
        return data_pos
