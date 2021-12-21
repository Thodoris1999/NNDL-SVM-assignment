
from sklearn.svm import SVC
import joblib
import argparse
import matplotlib.pyplot as plt

import utils
import viz_utils

def main(args):
    clf = joblib.load(args.model_name)
    train_images, train_labels, test_images, test_labels = utils.mnist()

    accuracy, precisions, recalls, cm, cm_examples, test_dur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    best_pipe = clf.best_estimator_
    best_pca = best_pipe['reduce_dim']
    best_svm = best_pipe['clf']
    print(f"PCA number of components variance: {best_pca.n_components_}")
    print(f"SVM number of support vectors per class: {best_svm.n_support_}")
    print(f"SVM accuracy: {accuracy}")
    print(f"SVM testing duration: {test_dur}")
    viz_utils.viz_cm_examples(cm_examples)
    viz_utils.viz_pca_components(best_pca)

    viz_utils.plot_cv_results_with_stdev(clf.cv_results_)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-m',default='svm.joblib')
    args = parser.parse_args()
    main(args)