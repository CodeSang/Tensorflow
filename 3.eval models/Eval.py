import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from utils.dataset_utils import load_batch
from datasets import tf_record_dataset
import numpy as np
import cv2

tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset prefix.')

tf.app.flags.DEFINE_string(
    'record_dir',
    None,
    'The name of the TFRecord prefix.')


tf.app.flags.DEFINE_string(
    'log',
    None,
    'log_check_File.')

tf.app.flags.DEFINE_integer(
    'num_classes',
    5,
    'The name of the TFRecord prefix.')

def run():
    tfrecord_dataset = tf_record_dataset.TFRecordDataset(
        tfrecord_dir= FLAGS.record_dir,
        dataset_name= FLAGS.dataset_name,
        num_classes= FLAGS.num_classes)

    dataset = tfrecord_dataset.get_split(split_name='validation')


    # Choice Model . vgg, inception.V1~3 , alexnet , resnet .. etc...



    inception = nets.inception
    X_image = tf.placeholder(tf.float32, shape=[None, inception.inception_v3.default_image_size,
                                                inception.inception_v3.default_image_size, 3])
    images, labels, _ = load_batch(dataset, height=inception.inception_v3.default_image_size,
                                   width=inception.inception_v3.default_image_size, num_classes = FLAGS.num_classes)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(inputs=X_image, num_classes=FLAGS.num_classes)

    predictions = tf.argmax(logits, 1)
    Y_label = tf.placeholder(tf.float32, shape=[None, 5])
    targets = tf.argmax( Y_label, 1)



    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)


    log_dir = FLAGS.log
    eval_dir = FLAGS.log
    if not tf.gfile.Exists(eval_dir):
        tf.gfile.MakeDirs(eval_dir)
    if not tf.gfile.Exists(log_dir):
        raise Exception("trained check point does not exist at %s " % log_dir)
    else:
        checkpoint_path = tf.train.latest_checkpoint(log_dir)

    import matplotlib.pyplot as plt
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        with slim.queues.QueueRunners(sess):

            for i in range(100):
                np_image, np_label = sess.run([images, labels])

                tempimage, tflabel = sess.run([predictions,targets], feed_dict={X_image:np_image,Y_label:np_label})

                print("Predict : " , tempimage)
                print("Answer  : ",  tflabel)
                print( 'enter' )




def main(_):
    if not FLAGS.dataset_name:
     raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.log:
     raise ValueError('You must supply the dataset name with --log')
    if not FLAGS.record_dir:
     raise ValueError('You must supply the dataset name with --record_dir')
    run()

if __name__ == '__main__':
    tf.app.run()

