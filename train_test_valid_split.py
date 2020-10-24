import os
import random
import shutil
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect


def choices(population, weights=None, *, cum_weights=None, k=1):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0    # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]


def split_dir(path, train_path, test_path, valid_path):
    random.seed(42)
    for name in os.listdir(path):
        dataset = choices(['train', 'test', 'valid'], [0.8, 0.15, 0.05])[0]
        dataset_path = None
        if dataset == 'train':
            dataset_path = train_path
        elif dataset == 'test':
            dataset_path = test_path
        elif dataset == 'valid':
            dataset_path = valid_path

        shutil.copyfile(os.path.join(path, name), os.path.join(dataset_path, name))


if __name__ == '__main__':
    cat_path = 'cat-vs-dog/cat'
    dog_path = 'cat-vs-dog/dog'
    valid_cat_path = 'cat-vs-dog/valid/cat'
    valid_dog_path = 'cat-vs-dog/valid/dog'
    test_cat_path = 'cat-vs-dog/test/cat'
    test_dog_path = 'cat-vs-dog/test/dog'
    train_cat_path = 'cat-vs-dog/train/cat'
    train_dog_path = 'cat-vs-dog/train/dog'

    os.makedirs(valid_cat_path)
    os.makedirs(valid_dog_path)
    os.makedirs(test_cat_path)
    os.makedirs(test_dog_path)
    os.makedirs(train_cat_path)
    os.makedirs(train_dog_path)

    split_dir(cat_path, train_cat_path, test_cat_path, valid_cat_path)
    print('Cat:')
    print('  Train: {} images'.format(len(os.listdir(train_cat_path))))
    print('  Test:  {} images'.format(len(os.listdir(test_cat_path))))
    print('  Valid: {} images'.format(len(os.listdir(valid_cat_path))))

    split_dir(dog_path, train_dog_path, test_dog_path, valid_dog_path)
    print('Dog:')
    print('  Train: {} images'.format(len(os.listdir(train_dog_path))))
    print('  Test:  {} images'.format(len(os.listdir(test_dog_path))))
    print('  Valid: {} images'.format(len(os.listdir(valid_dog_path))))

    shutil.rmtree(cat_path)
    shutil.rmtree(dog_path)
