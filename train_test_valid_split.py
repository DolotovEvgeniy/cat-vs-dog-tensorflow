import os
import random
import shutil

def split_dir(path, train_path, test_path, valid_path):
    random.seed(42)
    for name in os.listdir(path):
        dataset = random.choices(['train', 'test', 'valid'], [0.8, 0.15, 0.05])[0]
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
