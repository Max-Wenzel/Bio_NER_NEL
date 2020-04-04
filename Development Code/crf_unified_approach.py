import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score
from helper_functions import *


def unified_approach(file_name, file_2):
    # train_path = "../Data/bio-ner/train"
    # dev_path = "../Data/bio-ner/dev"
    # create_file(train_path, "train")
    # create_file(dev_path, "dev")
    #exclude = ["Value", "Time", "Unit", "Location"]
    train_sentences = file_opener(file_name)
    dev_sentences = file_opener(file_2)

    x_train = [sentence_features(s) for s in train_sentences]
    y_train = [sentence_labels(s) for s in train_sentences]

    x_dev = [sentence_features(s) for s in dev_sentences]
    y_dev = [sentence_labels(s) for s in dev_sentences]

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

    # Get the various lists for evaluation of separate label parts
    y_pred_flat = []
    y_pred_iob = []
    y_pred_class = []
    y_dev_flat = []
    y_dev_iob = []
    y_dev_class = []

    for x in y_predicted:
        y_pred_flat += x
        for xx in x:
            y_pred_iob.append(xx[0])
            if xx != 'O':
                y_pred_class.append(xx[2:])
            else:
                y_pred_class.append('O')
    for x in y_dev:
        y_dev_flat += x
        for xx in x:
            y_dev_iob.append(xx[0])
            if xx != 'O':
                y_dev_class.append(xx[2:])
            else:
                y_dev_class.append('O')

    # print(set(y_pred_flat) - set(y_dev_flat))
    # print(set(y_dev_flat) - set(y_pred_flat))
    # print(set(y_pred_flat))
    # print(set(y_dev_flat))
    # print(labels)
    labels = list(set(y_pred_flat))
    labels.remove("O")
    print(labels)
    #labels = ["B-Biotic_Entity-L"]
    f1 = metrics.flat_f1_score(y_dev, y_predicted, average='weighted', labels=labels)

    # labels = list(set(y_pred_iob))
    # labels.remove('O')
    # iob_score = f1_score(y_dev_iob, y_pred_iob, average='weighted', labels=labels)
    # print("IOB Score:", iob_score)
    # labels = list(set(y_pred_class))
    # labels.remove('O')
    # class_score = f1_score(y_dev_class, y_pred_class, average='weighted', labels=labels)
    # print("Class Score:", class_score)
    print("Overall Score:", f1)
    return f1


if __name__ == '__main__':
    unified_approach("./nel-labeled/train", "./nel-labeled/test")
