import os
import tensorflow as tf

def clean_dir(dir_path):
    print('Clean directory: ' + dir_path)
    count = 0
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)

        if tf.compat.as_bytes("JFIF") not in open(path, 'rb').peek(10):
            os.remove(path)
            count += 1
    print('{} images removed'.format(count))


if __name__ == '__main__':
    cat_path = os.path.join('cat-vs-dog/cat')
    dog_path = os.path.join('cat-vs-dog/dog')

    clean_dir(cat_path)
    clean_dir(dog_path)

    print('Done!')