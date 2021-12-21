
import argparse
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

import utils

def train_svm(X, y, params=None):
    pca_variance = 0.9
    pca = PCA(n_components=pca_variance, svd_solver='full')
    svm = SVC(cache_size=200)
    pipe = Pipeline([
        ('reduce_dim', pca), 
        ('clf', svm)
        ])
    if params is not None:
        pipe.set_params(**params)
    pipe.fit(X, y)
    return pipe

def main(args):
    train_images, train_labels, test_images, test_labels = utils.mnist()
    params = {"clf__kernel": "rbf", "clf__C": 10, "clf__gamma": 0.01}

    start = time.time()
    clf = train_svm(train_images, train_labels, params)
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