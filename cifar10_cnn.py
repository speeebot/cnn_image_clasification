from matplotlib import pyplot
from keras.datasets import cifar10
# load dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (train_x.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_x.shape, test_y.shape))
# plot first few images
#for i in range(9):
  # define subplot
#  pyplot.subplot(330 + 1 + i)
  # plot raw pixel data
#  pyplot.imshow(trainX[i])
# show the figure
#pyplot.show()

def main():
    epochs = 25
    batch_size = 128
    

if __name__ =  "__main__":
    main()
