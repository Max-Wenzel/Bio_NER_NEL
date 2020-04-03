import sklearn_crfsuite
from sklearn_crfsuite import metrics
from crf_funcs import *
import warnings
from simpletransformers.classification import ClassificationModel
from simpletransformers.ner import NERModel
import pandas as pd
from sklearn.metrics import f1_score
from copy import deepcopy


class BioAnalysis:
    def __init__(self, train_file="./data/train.tsv", dev_file="./data/dev.tsv", test_file="./data/test.tsv"):
        self.train_data = file_opener(train_file)
        self.dev_data = file_opener(dev_file)
        self.test_data = file_opener(test_file)
        self.test_data.pop(192)
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.09684573395986483,
            c2=0.0800864058815976,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.dev_predicted = None
        self.test_predicted = None
        self.dev_labels = None
        self.test_labels = None
        self.labels = ['B-Abiotic_Entity',
                       'B-Aggregate_Biotic_Abiotic_Entity',
                       'B-Biotic_Entity',
                       'B-Eventuality',
                       'B-Location',
                       'B-Quality',
                       'B-Time',
                       'B-Unit',
                       'B-Value',
                       'I-Abiotic_Entity',
                       'I-Aggregate_Biotic_Abiotic_Entity',
                       'I-Biotic_Entity',
                       'I-Eventuality',
                       'I-Location',
                       'I-Quality',
                       'I-Time',
                       'I-Unit',
                       'I-Value',
                       'O']

        self.roberta_nel_model = None
        self.roberta_nel_dev_eval = None
        self.roberta_nel_test_eval = None
        self.roberta_nel_dev_links = None
        self.roberta_nel_test_links = None
        self.roberta_nel_train_data, _ = get_roberta_nel_data(self.train_data)
        self.roberta_nel_dev_data, self.roberta_nel_dev_spans = get_roberta_nel_data(self.dev_data)
        self.roberta_nel_test_data, self.roberta_nel_test_spans = get_roberta_nel_data(self.test_data)

        self.roberta_ner_model = None
        self.roberta_ner_dev_eval = None
        self.roberta_ner_test_eval = None
        self.roberta_ner_train_data = get_roberta_ner_data(self.train_data)
        self.roberta_ner_dev_data = get_roberta_ner_data(self.dev_data)
        self.roberta_ner_test_data = get_roberta_ner_data(self.test_data)

    def crf_fit(self):
        self.crf.fit(*get_features_labels(self.train_data))

    def crf_predict(self):
        dev_feat, self.dev_labels = get_features_labels(self.dev_data)
        test_feat, self.test_labels = get_features_labels(self.test_data)
        self.dev_predicted = self.crf.predict(dev_feat)
        self.test_predicted = self.crf.predict(test_feat)

    def crf_evaluate(self, verbose=False, labels=False):
        if labels:
            lab = labels
        else:
            lab = self.crf.classes_
            lab.remove("O")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Dev Results\n===========")
            dev_args = (self.dev_labels, self.dev_predicted)
            kwargs = {"average": "weighted", "labels": lab}
            if verbose:
                print("Precision:", metrics.flat_precision_score(*dev_args, **kwargs))
                print("Recall:", metrics.flat_recall_score(*dev_args, **kwargs))
            print("F1:", metrics.flat_f1_score(*dev_args, **kwargs))
            test_args = (self.test_labels, self.test_predicted)
            print("\nTest Results\n============")
            if verbose:
                print("Precision:", metrics.flat_precision_score(*test_args, **kwargs))
                print("Recall:", metrics.flat_recall_score(*test_args, **kwargs))
            print("F1:", metrics.flat_f1_score(*test_args, **kwargs))

    def roberta_nel_train(self):
        train_df = pd.DataFrame(self.roberta_nel_train_data)
        self.roberta_nel_model = ClassificationModel("roberta", "roberta-base", args={"num_train_epochs": 3,
                                                                                      "overwrite_output_dir": True,
                                                                                      "output_dir": "nel_outputs/"})
        self.roberta_nel_model.train_model(train_df)

    def roberta_nel_eval(self):
        dev_df = pd.DataFrame(self.roberta_nel_dev_data)
        test_df = pd.DataFrame(self.roberta_nel_test_data)
        self.roberta_nel_dev_eval = self.roberta_nel_model.eval_model(dev_df, acc=f1_score)
        self.roberta_nel_test_eval = self.roberta_nel_model.eval_model(test_df, acc=f1_score)
        print("Dev NEL Results\n===========")
        print("F1:", self.roberta_nel_dev_eval[0]["acc"])
        print("\nTest NEL Results\n============")
        print("F1:", self.roberta_nel_test_eval[0]["acc"])

    def roberta_nel_load_model(self):
        self.roberta_nel_model = ClassificationModel("roberta", "nel_outputs/", args={"num_train_epochs": 3})

    def roberta_ner_train(self):
        train_df = pd.DataFrame(self.roberta_ner_train_data, columns=['sentence_id', 'words', 'labels'])
        self.roberta_ner_model = NERModel("roberta", "roberta-base", labels=self.labels,
                                          args={"num_train_epochs": 3, "overwrite_output_dir": True,
                                                "output_dir": "ner_outputs/"})
        self.roberta_ner_model.train_model(train_df)

    def roberta_ner_eval(self):
        dev_df = pd.DataFrame(self.roberta_ner_dev_data, columns=['sentence_id', 'words', 'labels'])
        test_df = pd.DataFrame(self.roberta_ner_test_data, columns=['sentence_id', 'words', 'labels'])
        self.roberta_ner_dev_eval = self.roberta_ner_model.eval_model(dev_df)
        self.roberta_ner_test_eval = self.roberta_ner_model.eval_model(test_df)
        print("Dev NER Results\n===========")
        print("Precision:", self.roberta_ner_dev_eval[0]["precision"])
        print("Recall:", self.roberta_ner_dev_eval[0]["recall"])
        print("F1:", self.roberta_ner_dev_eval[0]["f1_score"])
        print("\nTest NER Results\n============")
        print("Precision:", self.roberta_ner_test_eval[0]["precision"])
        print("Recall:", self.roberta_ner_test_eval[0]["recall"])
        print("F1:", self.roberta_ner_test_eval[0]["f1_score"])

    def roberta_ner_load_model(self):
        self.roberta_ner_model = NERModel("roberta", "ner_outputs/", labels=self.labels, args={"num_train_epochs": 3})

    def roberta_ner_nel_pipeline(self):
        self.roberta_ner_load_model()
        self.roberta_ner_eval()

        roberta_dev_phrases = deepcopy(self.dev_data)
        for ii in range(len(roberta_dev_phrases)):
            for jj in range(len(roberta_dev_phrases[ii])):
                roberta_dev_phrases[ii][jj] = list(roberta_dev_phrases[ii][jj])
                roberta_dev_phrases[ii][jj][2] = self.roberta_ner_dev_eval[2][ii][jj]
        roberta_dev_phrases, roberta_dev_spans = get_roberta_nel_data(roberta_dev_phrases)

        roberta_test_phrases = deepcopy(self.test_data)
        for ii in range(len(roberta_test_phrases)):
            for jj in range(len(roberta_test_phrases[ii])):
                roberta_test_phrases[ii][jj] = list(roberta_test_phrases[ii][jj])
                roberta_test_phrases[ii][jj][2] = self.roberta_ner_test_eval[2][ii][jj]
        roberta_test_phrases, roberta_test_spans = get_roberta_nel_data(roberta_test_phrases)

        self.roberta_nel_load_model()
        roberta_dev_prediction = ba.roberta_nel_model.predict([x[0] for x in roberta_dev_phrases])[0]
        roberta_test_prediction = ba.roberta_nel_model.predict([x[0] for x in roberta_test_phrases])[0]

        roberta_dev_actual = [x[1] for x in self.roberta_nel_dev_data]
        roberta_test_actual = [x[1] for x in self.roberta_nel_test_data]

        dev_prediction = transform_nel_results(roberta_dev_prediction, roberta_dev_spans)
        dev_actual = transform_nel_results(roberta_dev_actual, self.roberta_nel_dev_spans)
        dev_actual, dev_prediction = resolve_diff(dev_actual, dev_prediction)

        test_prediction = transform_nel_results(roberta_test_prediction, roberta_test_spans)
        test_actual = transform_nel_results(roberta_test_actual, self.roberta_nel_test_spans)
        test_actual, test_prediction = resolve_diff(test_actual, test_prediction)
        print("Dev NEL Combined Results\n===========")
        print("F1:", f1_score(dev_actual, dev_prediction))
        print("Test NEL Combined Results\n===========")
        print("F1:", f1_score(test_actual, test_prediction))

        dev_output = list(zip([x[0] for x in roberta_dev_phrases], roberta_dev_prediction))
        self.roberta_nel_dev_links = get_links(dev_output)
        test_output = list(zip([x[0] for x in roberta_test_phrases], roberta_test_prediction))
        self.roberta_nel_test_links = get_links(test_output)


if __name__ == '__main__':
    ba = BioAnalysis()
    ba.roberta_nel_load_model()
    ba.roberta_nel_eval()
