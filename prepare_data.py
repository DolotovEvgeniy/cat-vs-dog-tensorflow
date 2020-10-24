import tensorflow as tf

def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory('cat-vs-dog/train',
                                                        target_size=(224, 224),
                                                        color_mode="rgb",
                                                        batch_size=32,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    valid_generator = valid_datagen.flow_from_directory('cat-vs-dog/valid',
                                                        target_size=(224, 224),
                                                        color_mode="rgb",
                                                        batch_size=10,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    test_generator = test_datagen.flow_from_directory('cat-vs-dog/test',
                                                      target_size=(224, 224),
                                                      color_mode="rgb",
                                                      batch_size=10,
                                                      seed=7,
                                                      shuffle=True,
                                                      class_mode="categorical"
                                                      )


    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples


    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num
