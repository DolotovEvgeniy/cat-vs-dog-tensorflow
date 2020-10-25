import tensorflow as tf

def VGG16(l2_factor):
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu,
                                     input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 4
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 5
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                     activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2,
                                        padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                    activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=4096,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                    activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=2,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_factor),
                                    activation=tf.keras.activations.softmax))

    return model