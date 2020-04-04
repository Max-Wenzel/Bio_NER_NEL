import wikipedia
from nltk.stem import PorterStemmer
from random import shuffle

ps = PorterStemmer()


# help and inspiration from
# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-use-conll-2002-data-to-build-a-ner-system
# and
# https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2


def file_opener(file_name, exclude=False):
    sentence_list = []
    current_sentence = []
    with open(file_name) as f:
        for line in f:
            word = line.strip().split('\t')
            if word[0] == '_S_B_':
                if len(current_sentence) > 0:
                    sentence_list.append(current_sentence)
                    current_sentence = []
            else:
                if exclude:
                    if word[2][2:] not in exclude:
                        current_sentence.append(tuple(word))
                    else:
                        current_sentence.append(tuple([word[0], word[1], 'O']))
                else:
                    current_sentence.append(tuple(word))
    return sentence_list


def feature_extractor(sentence, i, step_two=False, predictions=False):
    word = sentence[i][0]
    pos = sentence[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'ps.stem(word)': ps.stem(word),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos': pos,
        'pos[:2]': pos[:2],
    }
    if step_two:
        if predictions:
            features.update({
                "IOB": predictions[i]
            })
        else:
            features.update({
                "IOB": sentence[i][2][0]
            })
    if i > 0:
        word1 = sentence[i - 1][0]
        pos1 = sentence[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:ps.stem(word)': ps.stem(word1),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:pos': pos1,
            '-1:pos[:2]': pos1[:2],
        })
        if step_two:
            if predictions:
                features.update({
                    "-1:IOB": predictions[i - 1]
                })
            else:
                features.update({
                    "-1:IOB": sentence[i - 1][2][0]
                })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        word1 = sentence[i + 1][0]
        pos1 = sentence[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:ps.stem(word)': ps.stem(word1),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:pos': pos1,
            '+1:pos[:2]': pos1[:2],
        })
        if step_two:
            if predictions:
                features.update({
                    "+1:IOB": predictions[i + 1]
                })
            else:
                features.update({
                    "+1:IOB": sentence[i + 1][2][0]
                })
    else:
        features['EOS'] = True

    return features


def remove_l(tag):
    if tag[-1] == "L":
        return tag[:-2]
    else:
        return tag


def sentence_features(sentence, step_two=False, predictions=False):
    return [feature_extractor(sentence, i, step_two=step_two, predictions=predictions) for i in range(len(sentence))]


def sentence_labels(sentence, step_one=False, step_two=False, ner=True):
    if step_one:
        return [label[0] for token, pos, label in sentence]
    elif step_two:
        return [label[2:] if label != 'O' else 'O' for token, pos, label in sentence]
    else:
        return [label[:-2] if label[-1] == 'L' and ner else label for token, pos, label in sentence]


def sentence_tokens(sentence):
    return [token for token, pos, label in sentence]


# This function as not used as I switched to a different way to do the same thing
def shuffle_data(a, b):
    comb = list(zip(a, b))
    shuffle(comb)
    comb = list(zip(*comb))
    return list(comb[0]), list(comb[1])


def get_features_labels(data, ner=True):
    feat = []
    labels = []
    for sent in data:
        feat.append(sentence_features(sent))
        labels.append(sentence_labels(sent, ner=ner))
    return feat, labels


def get_roberta_ner_data(data):
    sentences = []
    for ii in range(len(data)):
        for word in data[ii]:
            if word[2][-1] == "L":
                sentences.append([ii, word[0], word[2][:-2]])
            else:
                sentences.append([ii, word[0], word[2]])
    return sentences


def get_roberta_nel_data(data):
    chunks = []
    spans = []
    ind = 0
    for sentence in data:
        current = ["", None]
        in_flag = False
        for ii in range(len(sentence)):
            if in_flag:
                current[0] += sentence[ii][0] + " "
                if sentence[ii][2][0] != "I":
                    chunks.append(current)
                    in_flag[1] = ind
                    spans.append(in_flag)
                    current = ["", None]
                    in_flag = False

            if sentence[ii][2][:15] == "B-Biotic_Entity":
                in_flag = [ind, None]
                current[1] = int(sentence[ii][2][-1] == "L")
                if ii > 0:
                    current[0] = sentence[ii - 1][0] + " "
                else:
                    current[0] = "_S_B_ "
                current[0] += sentence[ii][0] + " "
            ind += 1
        if in_flag:
            current[0] += "_S_E_"
            chunks.append(current)
            in_flag[1] = ind
            spans.append(in_flag)
    return [chunks, spans]


def transform_nel_results(prediction, spans):
    out = [0] * spans[-1][1]
    pred_spans = list(zip(spans, prediction))
    for span in pred_spans:
        if span[1]:
            for ii in range(span[0][0], span[0][1]):
                out[ii] = 1
    return out


def resolve_diff(actual, prediction):
    diff = len(prediction) - len(actual)
    if diff > 0:
        actual += [0] * diff
    elif diff < 0:
        prediction += [0] * (diff * -1)
    return actual, prediction


def get_links(output):
    links = []
    for x in output:
        if x[1]:
            entity = " ".join(x[0].split()[1:-1])
            try:
                link = wikipedia.page(wikipedia.search(entity, results=1)[0]).url
                links.append(link)
            except:
                links.append(None)
    return links
