
from sklearn.svm import SVC
import joblib
import argparse

import utils

def main(args):
    clf = joblib.load(args.model_name)
    train_images, train_labels, test_images, test_labels = utils.mnist()

    accuracy, precisions, recalls, cm, test_dur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    print(f"SVM accuracy: {accuracy}")
    print(f"SVM training duration: {test_dur}")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name','-m',default='svm.joblib')
    args = parser.parse_args()
    main(args)