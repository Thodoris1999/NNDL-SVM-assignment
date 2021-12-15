
import argparse
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

import utils

def main(args):
    train_images, train_labels, test_images, test_labels = utils.mnist()

    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100, 1000]},
    ]
    start = time.time()
    clf = GridSearchCV(SVC(cache_size=200), tuned_parameters, scoring='accuracy')
    clf.fit(train_images, train_labels)
    optimization_dur = time.time()-start
    joblib.dump(clf, args.model_name)

    accuracy, precisions, recalls, cm, test_dur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    print(f"SVM accuracy: {accuracy}")
    print(f"SVM training duration: {optimization_dur}")
    print(f"SVM test duration: {test_dur}")
    print(f"SVM CV results:\n {clf.cv_results_}")
    print(f"SVM best parameters: {clf.best_params_}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-m',default='svm_gridsearch.joblib')
    args = parser.parse_args()
    main(args)
