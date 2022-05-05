import os
import pathlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 50

# load training & testing data
train_data_path = 'train_data'# Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
test_data_path = 'test_data'# Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
train_data_root = pathlib.Path(train_data_path)
test_data_root = pathlib.Path(test_data_path)

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.cast(tf.image.decode_jpeg(img, channels=1), tf.float32)
    return img


# list all training images paths，sort them to make the data and the label aligned
all_training_image_paths = list(train_data_root.glob('*'))
all_training_image_paths = sorted([str(path) for path in all_training_image_paths])
train_image_count = len(all_training_image_paths)
# list all testing images paths，sort them to make the data and the label aligned
all_testing_image_paths = list(test_data_root.glob('*'))
all_testing_image_paths = sorted([str(path) for path in all_testing_image_paths])
test_image_count = len(all_testing_image_paths)
# load labels for training and testing data
training_labels = np.loadtxt('labels/train_label.txt', dtype=int)
testing_labels = np.loadtxt('labels/test_label.txt', dtype=int)
# compile testing and training data
training_path_ds = tf.data.Dataset.from_tensor_slices(all_training_image_paths)
testing_path_ds = tf.data.Dataset.from_tensor_slices(all_testing_image_paths)
# training data
train_image_ds = training_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).batch(batch_size)
train_label = tf.data.Dataset.from_tensor_slices(tf.cast(training_labels, tf.int64)).batch(batch_size)
# testing data
test_image_ds = testing_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).batch(test_image_count)
test_label = tf.data.Dataset.from_tensor_slices(tf.cast(testing_labels, tf.int64)).batch(test_image_count)
# combine all the dataset
train_ds = tf.data.Dataset.zip((train_image_ds, train_label))
test_ds = tf.data.Dataset.zip((test_image_ds, test_label))
# preprocess and normalize the images
train_ds = train_ds.map(
    lambda x, y: (tf.divide(tf.reshape(x, (-1, 784)), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10)))).shuffle(
    train_image_count)
test_ds = test_ds.map(
    lambda x, y: (tf.divide(tf.reshape(x, (-1, 784)), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))

test_data, test_labels = zip(*test_ds)
X_test = test_data[0]
y_test = test_labels[0]
train_data, train_labels = zip(*train_ds)
X_train = train_data[0]
y_train = train_labels[0]
print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)

# idx = np.random.randint(0, len(X_train), size=1)[0]
# img1 = tf.reshape(X_train[idx], (28, 28)).numpy()
# img2 = tf.reshape(X_train[idx], (28, 28)).numpy()
# plt.imshow(img1)
# plt.imshow(img2)
# print(np.argmax(y_train[idx]))
# plt.show()

def initialize_weights():
    # initialize parameters to small random values
    minval = 0
    maxval = 0.001
    W1 = tf.Variable(tf.random.uniform([784, 100], minval, maxval, seed=0) , name='W1')
    W10 = tf.Variable(tf.random.uniform([100, 1], minval, maxval, seed=0) , name='W10')
    W2 = tf.Variable(tf.random.uniform([100, 100], minval, maxval, seed=1), name='W2')
    W20 = tf.Variable(tf.random.uniform([100, 1], minval, maxval, seed=1) , name='W20')
    W3 = tf.Variable(tf.random.uniform([100, 10], minval, maxval, seed=2) , name='W3')
    W30 = tf.Variable(tf.random.uniform([10, 1], minval, maxval, seed=2) , name='W30')

    return [W1, W10, W2, W20, W3, W30]

theta = initialize_weights()
print('W1 shape is : ',theta[0].shape)
print('W10 shape is : ',theta[1].shape)
print('W2 shape is : ',theta[2].shape)
print('W20 shape is : ',theta[3].shape)
print('W3 shape is : ',theta[4].shape)
print('W30 shape is : ',theta[5].shape)

def forward_propagation(X, theta):
    batch_size = X.shape[0]
    # first hidden layer H1
    Z1 = tf.matmul(X, theta[0]) + tf.matmul(tf.ones([batch_size, 1]), tf.transpose(theta[1]))
    H1 = tf.nn.relu(Z1)
    # second hidden layer H2
    Z2 = tf.matmul(H1, theta[2]) + tf.matmul(tf.ones([batch_size, 1]), tf.transpose(theta[3]))
    H2 = tf.nn.relu(Z2)
    # output layer y_hat
    Z3 = tf.matmul(H2, theta[4]) + tf.matmul(tf.ones([batch_size, 1]), tf.transpose(theta[5]))
    y_hat = tf.nn.softmax(Z3) + 1e-10

    return Z1, H1, Z2, H2, Z3, y_hat

def L1_regularization(lambd,theta):
    L1_reg = 0
    for i in range(0,5):
        L1_reg += lambd * tf.reduce_sum(tf.math.abs(theta[i]))
    return L1_reg.numpy()

def cross_entropy_loss(X, y, y_hat, theta, lambd):
    # cross-entropy loss
    loss = tf.reduce_mean(tf.reduce_sum(-tf.math.multiply(y, tf.math.log(y_hat)), axis=1))
    # add L1 regularization
    loss = loss + L1_regularization(lambd,theta)
    # return a numpy variable not a tensor
    return loss.numpy()

def accuracy(y, y_hat):
    # convert y_hat to one-hot vectors
    y_hat = tf.one_hot(tf.math.argmax(y_hat, 1), 10, dtype=tf.float32)
    # compute average accuracy
    avg_acc = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y, y_hat), axis=1))
    # classification error for each digit
    err = []
    for c in range(y.numpy().shape[1]):
        correct = tf.reduce_sum(tf.math.multiply(y_hat[:, c], y[:, c]))
        total = tf.reduce_sum(y[:, c])
        error = 1 - correct / total
        err.append(error)
    return avg_acc.numpy(), err

def back_propagation(X, y, theta):
    batch_size = X.shape[0]
    # comput y_hat through forward propagation
    Z1, H1, Z2, H2, Z3, y_hat = forward_propagation(X, theta)
    # gradient of loss w.r.t output y_hat
    # dy shape is (m,10)
    dy = - (y - y_hat) / batch_size
    # gradient of loss w.r.t W3 and W30
    # dW3 shape is (100,10)
    dW3 = tf.matmul(tf.transpose(H2), dy)
    # gradient of loss w.r.t W30
    # dW30 shape is (10,1)
    dW30 = tf.matmul(tf.transpose(dy), tf.ones([batch_size, 1]))
    # gradient of loss w.r.t H2
    # dH2 shape is (m,100)
    dH2 = tf.matmul(dy, tf.transpose(theta[4]))
    # gradient of loss w.r.t Z2
    # dZ2 shape is (m,100)
    dZ2 = tf.multiply(tf.cast(tf.greater(H2, 0), tf.float32), dH2)
    # gradient of loss w.r.t W2
    # dW2 shape is (100,100)
    dW2 = tf.matmul(tf.transpose(H1), dZ2)
    # gradient of loss w.r.t W20
    # dW20 shape is (100,1)
    dW20 = tf.matmul(tf.transpose(dZ2), tf.ones([batch_size, 1]))
    # gradient of loss w.r.t H1
    # dH1 shape is (m,100)
    dH1 = tf.matmul(dZ2, tf.transpose(theta[2]))
    # gradient of loss w.r.t Z1
    # dZ1 shape is (m,100)
    dZ1 = tf.multiply(tf.cast(tf.greater(H1, 0), tf.float32), dH1)
    # gradient of loss w.r.t W1
    # dW1 shape is (784,100)
    dW1 = tf.matmul(tf.transpose(X), dZ1)
    # gradient of loss w.r.t W10
    # dW10 shape is (100,1)
    dW10 = tf.matmul(tf.transpose(dZ1), tf.ones([batch_size, 1]))

    return [dW1, dW10, dW2, dW20, dW3, dW30]

dtheta = back_propagation(X_test, y_test, theta)
print('dW3 shape:', dtheta[4].shape)
print('dW30 shape:', dtheta[5].shape)
print('dW2 shape:', dtheta[2].shape)
print('dW20 shape:', dtheta[3].shape)
print('dW1 shape:', dtheta[0].shape)
print('dW10 shape:', dtheta[1].shape)

def model(X_train, y_train, X_test, y_test, theta, alpha, epochs, batch_size, lamd):
    # initialize weights to small random values
    theta = initialize_weights()
    # initialize training errors for plotting
    err_train = []
    acc_train = []
    err_digit_train = []
    # initialize training errors for plotting
    err_test = []
    acc_test = []
    err_digit_test = []
    average_error_test = []

    for epoch in range(epochs):

        # print epoch number
        print(f"\n\n Epoch: {epoch} \n\n")
        # compute loss and accuracy for test data
        Z1_test, H1_test, Z2_test, H2_test, Z3_test, y_hat_test = forward_propagation(X_test, theta)
        loss_test = cross_entropy_loss(X_test, y_test, y_hat_test, theta, lambd)
        avg_acc_test, err_digits_test = accuracy(y_test, y_hat_test)

        err_test.append(loss_test)
        acc_test.append(avg_acc_test)
        err_digit_test.append(err_digits_test)

        # plot of average test classification error
        average_err_test = tf.reduce_mean(err_digit_test)
        average_error_test.append(average_err_test)

        print(f"Testing loss: {loss_test} , Test accuracy: {avg_acc_test}")

        for X_train, y_train in train_ds:

            # compute loss and accuracy for training data
            Z1, H1, Z2, H2, Z3, y_hat_train = forward_propagation(X_train, theta)
            loss_train = cross_entropy_loss(X_train, y_train, y_hat_train, theta, lambd)
            avg_acc_train, err_digits_train = accuracy(y_train, y_hat_train)

            err_train.append(loss_train)
            acc_train.append(avg_acc_train)
            err_digit_train.append(err_digits_train)

            # compute gradients
            dtheta = back_propagation(X_train, y_train, theta)

            # update weights
            theta[0] = theta[0] - (alpha * dtheta[0])
            theta[1] = theta[1] - (alpha * dtheta[1])
            theta[2] = theta[2] - (alpha * dtheta[2])
            theta[3] = theta[3] - (alpha * dtheta[3])
            theta[4] = theta[4] - (alpha * dtheta[4])
            theta[5] = theta[5] - (alpha * dtheta[5])

            print(f"Training loss: {loss_train} , Training accuracy: {avg_acc_train}")

            if (avg_acc_test > 0.97):
                break
        if (avg_acc_test > 0.97):
            break

    # print final classification error for each digit
    for i in range(0, 9):
        print(f"Classification error for digit {i} is: {err_digits_train[i]}")

    # print final training and testing losses and accuracies
    print(f"Final training loss: {loss_train + 1e-10} , Final training accuracy: {avg_acc_train * 100}")
    print(f"Final test loss: {loss_test + 1e-10} , Final testing accuracy: {avg_acc_test * 100}")

    # save parameters
    filehandler = open("nn_parameters.txt", "wb")
    pickle.dump(theta, filehandler, protocol=2)
    filehandler.close()

    return err_train, acc_train, err_digit_train, err_test, acc_test, err_digit_test, average_error_test

# initialize the hyper-parameters
alpha = 0.4         # learning rate
epochs = 20         # number of epochs
lambd = 1e-5        # regularization constant
batch_size = 50     # batch size
err_train, acc_train, err_digit_train, err_test, acc_test, err_digit_test, average_error_test = model(X_train, y_train, X_test, y_test, theta, alpha, epochs, batch_size, lambd)

# plot training loss
plt.figure(1)
plt.figure(figsize=(8, 8))
plt.plot(err_train, label=f'Training Loss (a={alpha})')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Loss Curve')
#plt.yscale('log')
plt.xlabel('# iterations')
plt.show()

# plot testing loss
plt.figure(2)
plt.figure(figsize=(8, 8))
plt.plot(err_test, label=f'Testing Loss (a={alpha} and {epochs} epochs)')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.xlabel('# epochs')
plt.show()

# plot training accuracy
plt.figure(3)
plt.figure(figsize=(8, 8))
plt.plot(acc_train, label=f'Training accuracy (a={alpha} and {epochs} epochs)')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('# iterations')
#plt.yscale('log')
plt.show()

# plot testing accuracy
plt.figure(4)
plt.figure(figsize=(8, 8))
plt.plot(acc_test, label=f'Testing accuracy (a={alpha} and {epochs} epochs)')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('# epochs')
# plt.yscale('log')
plt.show()

# plot testing err_each_digit
plt.figure(4)
plt.figure(figsize=(8, 8))
plt.plot(err_digit_test)
plt.legend(['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7','Digit 8', 'Digit 9'], loc='upper right')
plt.ylabel('Error')
plt.title('Testing Classification error for each digit')
plt.xlabel('# epochs')
# plt.yscale('log')
plt.show()

# plot final average test classification error
plt.figure(5)
plt.figure(figsize=(8, 8))
plt.plot(average_error_test, label=f'Average test classification error (a={alpha})')
plt.legend(loc='upper right')
plt.ylabel('Error')
plt.title('Loss Curve')
#plt.yscale('log')
plt.xlabel('# epochs')
plt.show()
