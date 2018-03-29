
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from utils.dataset_utils import load_batch
from datasets import tf_record_dataset


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

tf.app.flags.DEFINE_integer(
    'step',
    10000,
    'number_of_steps')


tf.app.flags.DEFINE_float(
    'learning_rate',
    0.001,
    'learning_rate')


def run():
    tfrecord_dataset = tf_record_dataset.TFRecordDataset(
        tfrecord_dir= FLAGS.record_dir,
        dataset_name= FLAGS.dataset_name,
        num_classes= FLAGS.num_classes)

    dataset = tfrecord_dataset.get_split(split_name='train')


    # Choice Model . vgg, inception.V1~3 , alexnet , resnet .. etc...

    inception = nets.inception

    images, labels, _ = load_batch(dataset, height=inception.inception_v3.default_image_size, width=inception.inception_v3.default_image_size, num_classes = FLAGS.num_classes)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(inputs=images, num_classes=FLAGS.num_classes)

    #loss Function
    loss = slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = slim.losses.get_total_loss()

    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    predictions = tf.argmax(logits, 1)
    targets = tf.argmax(labels, 1)

    correct_prediction = tf.equal(predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('losses/Total', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    #
    log_dir = FLAGS.log
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    # 훈련 오퍼레이션 정의
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(
        train_op,
        log_dir,
        number_of_steps=FLAGS.step,
        summary_op=summary_op,
        save_summaries_secs=30,
        save_interval_secs=30)


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
