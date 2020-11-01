import tensorflow as tf
from models.vgg16 import VGG16
from models.alexnet import AlexNet
from prepare_data import get_datasets

BATCH_SIZE = 32

def get_model(l2_factor):
    model = AlexNet(l2_factor=l2_factor)
    #model = VGG16(l2_factor=l2_factor)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.3)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    model = get_model(l2_factor=0.0001)
    model.summary()

    # start training
    model.fit(x=train_generator,
              epochs=100,
              steps_per_epoch=train_num // BATCH_SIZE,
              validation_data=valid_generator,
              validation_steps=valid_num // 10,
              callbacks=callback_list,
              workers=4,
              use_multiprocessing=True)

    # save the whole model
    model.save('model.h5')
