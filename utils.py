
import gzip
import os
import time
from urllib.request import urlretrieve
import numpy as np
from sklearn.metrics import confusion_matrix


def eval_sklearn_clf(clf, X, y):
    size = len(y)
    start = time.time()
    pred = clf.predict(X)
    dur = time.time()-start

    # Find examples of (mis)classification for each combination
    cm_examples = np.zeros((10,10,28,28))
    for x,yi,predi in zip(X,y,pred):
        cm_examples[int(yi), int(predi)] = np.reshape(x, (28,28))
    
    cm = confusion_matrix(y, pred)
    precisions = np.diag(cm) / np.sum(cm, axis=1)
    recalls = np.diag(cm) / np.sum(cm, axis=0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    ncorr = sum([1 for y_pred, label in zip(pred, y) if y_pred == label])
    return accuracy, precisions, recalls, cm, cm_examples, dur


#credit: https://mattpetersen.github.io/load-mnist-with-numpy
"""Load from /home/USER/data/mnist or elsewhere; download if missing."""
def mnist(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to ./data/MNIST/raw (common to pytorch default dataloader path)
        path = os.path.join('data', 'MNIST', 'raw')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        return integer_labels

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels