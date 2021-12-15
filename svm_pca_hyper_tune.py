
import argparse
import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import joblib

import utils
import viz_utils

def main(args):
    train_images, train_labels, test_images, test_labels = utils.mnist()

    # Retain 90% of the variance https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca_variance = 0.9
    pca = PCA(n_components=pca_variance, svd_solver='full')
    svm = SVC(cache_size=200)
    pipe = Pipeline([
        ('reduce_dim', pca), 
        ('clf', svm)
        ])

    tuned_parameters = [
        {"clf__kernel": ["rbf"], "clf__gamma": [1e-1, 1e-2, 1e-3, 1e-4], "clf__C": [1, 10, 100, 1000]},
        {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1, 10, 100, 1000]},
    ]
    start = time.time()
    clf = GridSearchCV(pipe, tuned_parameters, scoring='accuracy')
    clf.fit(train_images, train_labels)
    optimization_dur = time.time()-start
    joblib.dump(clf, args.model_name)

    accuracy, precisions, recalls, cm, test_dur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    best_pipe = clf.best_estimator_
    best_pca = best_pipe['reduce_dim']
    print(f"PCA number of components at {100*pca_variance}% variance: {best_pca.n_components_}")
    print(f"SVM accuracy: {accuracy}")
    print(f"SVM training duration: {optimization_dur}")
    print(f"SVM test duration: {test_dur}")
    print(f"SVM CV results:\n {clf.cv_results_}")
    print(f"SVM best parameters: {clf.best_params_}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-m',default='svm_pca_gridsearch.joblib')
    args = parser.parse_args()
    main(args)
