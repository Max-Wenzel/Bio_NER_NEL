import sklearn_crfsuite
from sklearn_crfsuite import metrics
from helper_functions import *


def step_one(file_name, file_2):
    train_sentences = file_opener(file_name)
    dev_sentences = file_opener(file_2)

    x_train = [sentence_features(s) for s in train_sentences]
    y_train = [sentence_labels(s, step_one=True) for s in train_sentences]

    x_dev = [sentence_features(s) for s in dev_sentences]
    y_dev = [sentence_labels(s, step_one=True) for s in dev_sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.09684573395986483,
        c2=0.0800864058815976,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    labels = list(crf.classes_)
    labels.remove('O')
    y_predicted = crf.predict(x_dev)

    f1 = metrics.flat_f1_score(y_dev, y_predicted, average='weighted', labels=labels)
    print("IOB Score:", f1)
    return step_two(train_sentences, dev_sentences, y_predicted)


def step_two(train_sentences, dev_sentences, y_predicted_iob):
    x_train = [sentence_features(s, step_two=True) for s in train_sentences]
    y_train = [sentence_labels(s, step_two=True) for s in train_sentences]

    x_dev = []
    for ii in range(len(dev_sentences)):
        x_dev.append(sentence_features(dev_sentences[ii], step_two=True, predictions=y_predicted_iob[ii]))
    y_dev = [sentence_labels(s, step_two=True) for s in dev_sentences]
    dev_key = [sentence_labels(s) for s in dev_sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.09684573395986483,
        c2=0.0800864058815976,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(x_train, y_train)
    labels = list(crf.classes_)
    labels.remove('O')
    print(labels)
    y_predicted = crf.predict(x_dev)

    f1 = metrics.flat_f1_score(y_dev, y_predicted, average='weighted', labels=labels)
    print("Class Score:", f1)

    combined = []
    for ii in range(len(y_predicted)):
        combo = list(zip(y_predicted_iob[ii], y_predicted[ii]))
        combined.append(list(map(lambda j: j[0] + "-" + j[1] if j[0] != 'O' else 'O', combo)))

    y_pred_flat = []
    for x in dev_key:
        y_pred_flat += x

    labels = list(set(y_pred_flat))
    labels.remove('O')

    final_f1 = metrics.flat_f1_score(combined, dev_key, average='weighted', labels=labels)
    print("Overall Score:", final_f1)


if __name__ == '__main__':
    step_one("./nel-labeled/train", "./nel-labeled/dev")
