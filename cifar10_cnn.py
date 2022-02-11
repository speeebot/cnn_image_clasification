import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from keras.datasets import cifar10

def getData():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x, test_x = train_x / 255.0, test_x / 255.0
    return train_x, train_y, test_x, test_y

def main():
    epochs = 25
    batch_size = 128

    # load dataset
    train_x, train_y, test_x, test_y = getData();

    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (train_x.shape, train_y.shape))
    print('Test: X=%s, y=%s' % (test_x.shape, test_y.shape))

    #create model
    model = tf.keras.models.Sequential([
        #tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation = "relu"),
        tf.keras.layers.Dense(2048, activation = "relu"),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    #configure model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],)

    #print out model summary
    model.summary()

    #train model
    model.fit(train_x, train_y, batch_size, epochs)

    #evaluate on testing data
    model.evaluate(test_x, test_y)

if __name__ ==  "__main__":
    main()
