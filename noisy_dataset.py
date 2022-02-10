from __future__ import print_function
import os
import os.path
import hashlib
import errno
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric' or noise_type == 'sn':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

class DatasetWrapper(torch.utils.data.Dataset):
    """Noise Dataset Wrapper"""

    def __init__(self, dataset, noise_type='clean', noise_rate=0,
                 yfile=None, weights_file=None, noise_train=False,
                 only_labeled=False, num_cls=10):
        """

        Args:
            dataset: the dataset to wrap, it should be an classification dataset
            noise_type: how to add noise for label: [clean/symmetric/asymmetric]
            noise_rate: noise ratio of adding noise
            yfile: The directory for the "y.npy" file. Once yfile assigned, we
                   will load yfile as labels and the given noise option will be
                   neglect. The weight of each sample will set to an binary
                   value according to the matching result of origin labels.
            weights_file: The weights for each samples, it should be an .npy
                   file of shape [len(dataset)] with either binary value or
                   probability value between [0,1]. "Specifically, all of the
                   unlabeled data should have zero-weight." The loaded weights
                   will multiply with the exists noise_or_not. So, it's ok to
                   give an weights for labeled data (noisy or clean).
        """
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_cls = num_cls

        if yfile is not None:
            yy = np.load(yfile)
            assert len(yy) == len(dataset)
            self.labels_to_use = yy
            # give zero-weights for incorrect sample
            self.weights = (self.labels_to_use == np.asarray(dataset.targets))
            self.noise_rate = 1 - (np.sum(self.weights) / len(self.weights))
            self.noise_type = "preload"
        elif noise_type == "clean":
            self.weights = np.ones(len(dataset))
            self.labels_to_use = dataset.targets
        else:
            # noisify labels
            train_clean_labels = np.expand_dims(np.asarray(dataset.targets), 1)
            train_noisy_labels, _ = noisify(train_labels=train_clean_labels,
                                            nb_classes=self.num_cls,
                                            noise_type=noise_type,
                                            noise_rate=noise_rate)
            self.labels_to_use = train_noisy_labels.flatten()
            assert len(self.labels_to_use) == len(dataset.targets)
            self.weights = (np.transpose(self.labels_to_use) ==
                            np.transpose(train_clean_labels)).squeeze()

        if noise_train:
            self.weights = np.ones(len(dataset))

        if weights_file is not None:
            # weights_file can be weights.npy or labeled.npy
            assert self.noise_type in ['preload', 'clean']
            self.useit = np.load(weights_file)
            assert len(self.useit) == len(dataset)
            if self.useit.dtype == np.bool:
                self.useit = self.useit.astype(np.float)
            self.weights = self.weights * self.useit

        if only_labeled:
            print("Removing unlabeled data for training efficiency...")
            origin_targets = np.asarray(dataset.targets)
            origin_data = dataset.data
            new_targets = origin_targets[self.weights != 0]
            new_data = origin_data[self.weights != 0]
            dataset.targets = new_targets
            dataset.data = new_data
            self.labels_to_use = np.asarray(self.labels_to_use)
            self.labels_to_use = self.labels_to_use[self.weights != 0]
            if weights_file is not None:
                self.useit = self.useit[self.weights != 0]
            self.weights = self.weights[self.weights != 0]
            print("Removed {} data with 0 weights!!!".format(
                len(origin_targets)-len(new_targets)))

    def save_noise_labels(self, dir):
        np.save(dir, np.asarray(self.labels_to_use))

    def __getitem__(self, index):
        # self.noise_or_not can expand to the weights of sample. So we can load
        # Semi-Supervised dataset here.
        img, target_gt = self.dataset[index]
        target_use = self.labels_to_use[index]
        weights = self.weights[index]
        # return img, target_use, target_gt, weights
        return img, target_use

    def __len__(self):
        return len(self.dataset)