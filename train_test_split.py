import os
import random
import shutil

def split_dir(path, train_path, test_path):
    random.seed(42)
    for name in os.listdir(path):
        if random.choices(['train', 'test'], [0.9, 0.1])[0] == 'test':
            shutil.copyfile(os.path.join(path, name), os.path.join(test_path, name))
        else:
            shutil.copyfile(os.path.join(path, name), os.path.join(train_path, name))


if __name__ == '__main__':
    cat_path = 'cat-vs-dog/cat'
    dog_path = 'cat-vs-dog/dog'
    test_cat_path = 'cat-vs-dog/test/cat'
    test_dog_path = 'cat-vs-dog/test/dog'
    train_cat_path = 'cat-vs-dog/train/cat'
    train_dog_path = 'cat-vs-dog/train/dog'

    os.makedirs(test_cat_path)
    os.makedirs(test_dog_path)
    os.makedirs(train_cat_path)
    os.makedirs(train_dog_path)

    split_dir(cat_path, train_cat_path, test_cat_path)
    print('Cat:')
    print('  Train: {} images'.format(len(os.listdir(train_cat_path))))
    print('  Test: {} images'.format(len(os.listdir(test_cat_path))))

    split_dir(dog_path, train_dog_path, test_dog_path)
    print('Dog:')
    print('  Train: {} images'.format(len(os.listdir(train_dog_path))))
    print('  Test: {} images'.format(len(os.listdir(test_dog_path))))

    shutil.rmtree(cat_path)
    shutil.rmtree(dog_path)
