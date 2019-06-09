import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import io
import random
import numpy as np
from tensorflow.contrib import rnn


def mean_var_norm(X):
    std_scaler = StandardScaler()
    scaled_X = std_scaler.fit_transform(X)
    #print(scaled_X.mean(axis=0))
    #print(scaled_X.std(axis=0))
    return scaled_X


def initialize_params():
    params = {}

    ## input to 1st hidden
    params['W1'] = tf.Variable(tf.truncated_normal([60, 40], stddev=0.01, dtype=tf.float32))
    params['b1'] = tf.Variable(tf.constant(1.0, shape=[40]))

    ## 1st to 2nd hidden
    params['W2'] = tf.Variable(tf.truncated_normal([40, 10], stddev=0.01, dtype=tf.float32))
    params['b2'] = tf.Variable(tf.constant(1.0, shape=[10]))

    params['Wout'] = tf.Variable(tf.truncated_normal([10, 1], stddev=0.01, dtype=tf.float32))
    params['bout'] = tf.Variable(tf.constant(1.0, shape=[1]))

    return params


def model(x, params):
    Y_flattened = tf.nn.relu(tf.matmul(x, params['W1']) + params['b1'])

    Y_flattened = tf.nn.relu(tf.matmul(Y_flattened, params['W2']) + params['b2'])

    output = tf.tanh((tf.matmul(Y_flattened, params['Wout']) + params['bout']))

    return output


def next_batch(X, y, batch_size, counter=0):
    start = counter
    while True:
        end = start + batch_size
        counter += batch_size

        batch_x = X[start:end, :]
        batch_y = y[start:end, :]

        yield batch_x, batch_y


def train_neural_network(X_mat, Y_mat, alpha, num_epochs, ckpt_dir, batch_size=256, shuffle_data=True):
    random.seed(random.randint(0, 10000))
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 60])
    Y = tf.placeholder(tf.float32, [None, 1])  # 1 output label open
    parameters = initialize_params()
    prediction = model(X, parameters)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
    optimizer = tf.train.RMSPropOptimizer(alpha).minimize(cost)
    epoch_count = 0
    with tf.name_scope("calc_loss"):
        calc_loss = tf.placeholder('float', name="calc_loss")
    with tf.name_scope("batch_L"):
        batch_L = tf.placeholder('float', name="batch_L")
    with tf.name_scope("overall"):
        overall = tf.truediv(calc_loss, batch_L, "overall")

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, "losslog"), sess.graph)
    overall_loss = 0
    tf.summary.scalar("calc_loss", calc_loss)
    tf.summary.scalar("batch_L", batch_L)
    tf.summary.scalar("overall", overall)
    tf.summary.scalar("loss", cost)
    summ_op = tf.summary.merge_all()
    X_train, X_val, y_train, y_val = train_test_split(X_mat, Y_mat, test_size=0.1, random_state=1)
    print(X_val.shape[0], " number of frames are used for validation")
    num_batches = X_train.shape[0] // batch_size
    L = 1
    val_loss = []

    for epoch in range(num_epochs):
        start_time = time.time()
        counter = 0
        X_train, y_train = shuffle(X_train, y_train, random_state=0)

        gen_obj = next_batch(X_train, y_train, batch_size)
        for i in range(num_batches):
            # x_batch,y_batch = next_batch(X_train,y_train,batch_size)
            try:
                x_batch, y_batch = next(gen_obj)
            except IndexError:
                break
            number_of_frames = x_batch.shape[0]
            assert x_batch.shape[0] == y_batch.shape[0]
            over = sess.run(overall, feed_dict={calc_loss: float(overall_loss), batch_L: float(L)})
            summ, minibatchloss, summ = sess.run([optimizer, cost, summ_op], feed_dict={X: x_batch, Y: y_batch,
                                                                                        calc_loss: float(overall_loss),
                                                                                        batch_L: float(L)})
            summary_writer.add_summary(summ, (epoch))

            end_time = time.time()
            overall_loss += minibatchloss
            L += 1
        print("Epoch ", epoch + 1, "Finishes", "Training loss:", overall_loss / L, "Time:", (end_time - start_time))
        if (epoch % 5) == 0:
            saver.save(sess, os.path.join(ckpt_dir, "mymodel.ckpt"))
            print("The model parameters have been put in a checkpoint for epoch ", epoch)
            epoch_count = epoch_count + 1
        val_loss.append(sess.run(cost, feed_dict={X: X_val, Y: y_val}))
    print("Validation error: ", val_loss)
    saver.save(sess, os.path.join(ckpt_dir, "mymodel.ckpt"))
    sess.close()
    print("The model parameters are saved")
    return val_loss


def RNN(x, params, timesteps):
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    # hidden units hard-coded for now
    lstm_cell = rnn.BasicLSTMCell(10, forget_bias=1.0)

    # Get lstm cell output
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], params['Wout']) + params['bout']


def initialize_rnn_params():
    params = {}

    params['Wout'] = tf.Variable(tf.truncated_normal([10, 1], stddev=0.01, dtype=tf.float32))
    params['bout'] = tf.Variable(tf.constant(1.0, shape=[1]))

    return params


def train_rnn(X_mat, Y_mat, alpha, num_epochs, timesteps, ckpt_dir, batch_size=256, shuffle_data=True):
    random.seed(random.randint(0, 10000))
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, timesteps, 1])
    Y = tf.placeholder(tf.float32, [None, 1])  # 1 output label open

    parameters = initialize_params()
    prediction = RNN(X, parameters, timesteps)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
    optimizer = tf.train.RMSPropOptimizer(alpha).minimize(cost)
    epoch_count = 0
    with tf.name_scope("calc_loss"):
        calc_loss = tf.placeholder('float', name="calc_loss")
    with tf.name_scope("batch_L"):
        batch_L = tf.placeholder('float', name="batch_L")
    with tf.name_scope("overall"):
        overall = tf.truediv(calc_loss, batch_L, "overall")

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, "losslog"), sess.graph)
    overall_loss = 0
    tf.summary.scalar("calc_loss", calc_loss)
    tf.summary.scalar("batch_L", batch_L)
    tf.summary.scalar("overall", overall)
    tf.summary.scalar("loss", cost)
    summ_op = tf.summary.merge_all()
    X_train, X_val, y_train, y_val = train_test_split(X_mat, Y_mat, test_size=0.1, random_state=1)
    print(X_val.shape[0], " number of frames are used for validation")
    num_batches = X_train.shape[0] // batch_size
    L = 1
    val_loss = []
    for epoch in range(num_epochs):
        start_time = time.time()
        counter = 0
        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_val = X_val.reshape((X_val.shape[0], timesteps, 1))
        gen_obj = next_batch(X_train, y_train, batch_size)
        for i in range(num_batches):
            # x_batch,y_batch = next_batch(X_train,y_train,batch_size)
            try:
                x_batch, y_batch = next(gen_obj)
            except IndexError:
                break
            assert x_batch.shape[0] == y_batch.shape[0]
            x_batch = x_batch.reshape((batch_size, 60, 1))
            over = sess.run(overall, feed_dict={calc_loss: float(overall_loss), batch_L: float(L)})
            summ, minibatchloss, summ = sess.run([optimizer, cost, summ_op], feed_dict={X: x_batch, Y: y_batch,
                                                                                        calc_loss: float(overall_loss),
                                                                                        batch_L: float(L)})
            summary_writer.add_summary(summ, (epoch))

            end_time = time.time()
            overall_loss += minibatchloss
            L += 1
        print("Epoch ", epoch + 1, "Finishes", "Training loss:", overall_loss / L, "Time:", (end_time - start_time))
        if (epoch % 5) == 0:
            saver.save(sess, os.path.join(ckpt_dir, "mymodel.ckpt"))
            print("The model parameters have been put in a checkpoint for epoch ", epoch)
            epoch_count = epoch_count + 1
        val_loss.append(sess.run(cost, feed_dict={X: X_val, Y: y_val}))
    print("Validation error: ", val_loss)
    saver.save(sess, os.path.join(ckpt_dir, "mymodel.ckpt"))
    sess.close()
    print("The model parameters are saved")
    return val_loss




def get_all_frames(all_binaries, inp_dim, total_frames):
    data_X = np.empty((total_frames,inp_dim))
    frames_current = 0
    for i,feat_file in enumerate(all_binaries):
        data , number_of_frames_in_this_file = io.binary_to_array_read(os.path.join('mfcc',feat_file), inp_dim)
        assert number_of_frames_in_this_file > 0
        if frames_current < total_frames:
            data_X[frames_current:frames_current+number_of_frames_in_this_file] = data[:number_of_frames_in_this_file]
            frames_current = frames_current + number_of_frames_in_this_file
        else:
            break
    return data_X



def plot_validation(val):
    plt.style.use('seaborn-paper')

    plt.plot(val)

    plt.xlabel('iterations')

    plt.xlim([0, len(val) - 1])

    plt.ylabel('Validation error')

    plt.show()



