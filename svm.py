
import argparse
import time
from sklearn.svm import SVC
import numpy as np
import joblib

import utils

def train_svm(X, y, params=None):
    clf = SVC()
    if params is not None:
        clf.set_params(params)
    clf.fit(X, y)
    return clf

def main(args):
    train_images, train_labels, test_images, test_labels = utils.mnist()

    start = time.time()
    clf = train_svm(train_images, train_labels)
    train_dur = time.time()-start
    joblib.dump(clf, args.model_name)

    accuracy, precisions, recalls, cm, cm_examples, test_dur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    print(f"SVM accuracy: {accuracy}")
    print(f"SVM training duration: {train_dur}")
    print(f"SVM test duration: {test_dur}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-m',default='svm.joblib')
    args = parser.parse_args()
    main(args)