import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def get_data():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    #normalize to range 0-1
    train_x, test_x = train_x / 255.0, test_x / 255.0
    return train_x, train_y, test_x, test_y

def plot_results(history, metric, val_metric):
    fig, axs = plt.subplots(2)
    fig.suptitle('Results')
    axs[0].plot(history.history[metric])
    axs[0].plot(history.history[val_metric])
    axs[0].set_title('model accuracy')
    axs[0].set(xlabel='epoch', ylabel='accuracy')
    axs[0].legend(['train', 'test'], loc='upper left')
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set(xlabel='epoch', ylabel='loss')
    axs[1].legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    epochs = 1
    batch_size = 128
    opt = tf.keras.optimizers.Adam(0.001)
    loss_fun = 'sparse_categorical_crossentropy'
    metric = 'sparse_categorical_accuracy'
    val_metric = 'val_sparse_categorical_accuracy'

    # load dataset
    train_x, train_y, test_x, test_y = get_data();
    #train_y = to_categorical(train_y)
    #test_y = to_categorical(test_y)

    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (train_x.shape, train_y.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, test_y.shape))

    #create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(2048, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(2048, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation = "relu"),
  #      tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation = "relu"),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    #configure model
    model.compile(optimizer=opt, loss=loss_fun, metrics=[metric])

    #print out model summary
    model.summary()

    #train model
    history = model.fit(train_x, train_y, batch_size, epochs, validation_split=0.20)

    #evaluate on testing data
    print("Evaluating on testing data:")
    test_loss, test_acc = model.evaluate(test_x, test_y)

    print("Test accuracy: {}".format(test_acc))

    #display accuracy over epochs
    plot_results(history, metric, val_metric)

if __name__ ==  "__main__":
    main()
