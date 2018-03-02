import tensorflow as tf
import numpy as np
import cv2

filename_queue = tf.train.string_input_producer(['output.csv'],shuffle=False, name = 'filename_queue')

batchSize = 10
training_epochs = 100
learning_rate = 0.001

ImageWidth = 256
ImageHeight = 256
yClass = 360

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_default = [[''], [0.]]
xy = tf.decode_csv(value, record_defaults = record_default)

train_x_batch, train_y_batch = tf.train.batch([xy[0],xy[1]],batch_size = batchSize)

sess = tf.Session()

#Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord=coord)



class Model:

    def __init__(self, sess, name , IWidth , IHeight, YClass):
        self.sess = sess
        self.name = name

        self.width = IWidth
        self.height  = IHeight
        self.yClass = YClass

        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.width * self.height])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1,  self.width, self.height, 1])
            self.Y = tf.placeholder(tf.float32, [None, self.yClass])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 32 * 32])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1000, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout4, units= self.yClass)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

m1 = Model(sess, "m1", ImageWidth, ImageHeight, yClass)
sess.run(tf.global_variables_initializer())

print('Learning Started!')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(360 / batchSize)

    for i in range(total_batch):
        batch_t_xs, batch_t_ys = sess.run([train_x_batch, train_y_batch])
        batch_xs = []
        batch_ys = []

        for temp in batch_t_xs:
            img = cv2.imread(temp.decode("utf-8"), 0)
            img2 = cv2.resize(img, (ImageWidth, ImageHeight))
            batch_xs.append(np.reshape(img2, ImageHeight * ImageWidth))

        for yt in batch_t_ys:
            temp = [0] * yClass
            temp[int(yt)] = 1.
            batch_ys.append(temp)

        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy


total_batch = int(360 / batchSize)

for i in range(total_batch):
    batch_t_xs, batch_t_ys = sess.run([train_x_batch, train_y_batch])
    batch_xs = []
    batch_ys = []

    for temp in batch_t_xs:
        img = cv2.imread(temp.decode("utf-8"), 0)
        img2 = cv2.resize(img, (ImageWidth, ImageHeight))
        batch_xs.append(np.reshape(img2, ImageHeight * ImageWidth))

    for yt in batch_t_ys:
        temp = [0] * yClass
        temp[int(yt)] = 1.
        batch_ys.append(temp)

    c = m1.get_accuracy(batch_xs, batch_ys)
    avg_cost += c / float(total_batch)

print('Accuracy:',avg_cost)


coord.request_stop()
coord.join(threads)

