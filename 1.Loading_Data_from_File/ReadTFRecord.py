import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from datasets import convert_tf_record
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


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.record_dir:
        raise ValueError('You must supply the TFRecord_Dir name with --record_dir')

    data_set_dummy = tf_record_dataset.TFRecordDataset(tfrecord_dir=FLAGS.record_dir,
                                                dataset_name=FLAGS.dataset_name,
                                                num_classes=10)

    dataset = data_set_dummy.get_split(split_name='train')

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            plt.figure()
            for i in range(4):
                np_image, np_label = sess.run([image, label])
                height, width, _ = np_image.shape
                class_name = name = dataset.labels_to_names[np_label]

                plt.subplot(2, 2, i+1)
                plt.imshow(np_image)
                plt.title('%d, %s, %d x %d' % (np_label,name, height, width))
                plt.axis('off')
            plt.show()

if __name__ == '__main__':
    tf.app.run()