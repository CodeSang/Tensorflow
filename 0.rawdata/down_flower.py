
#for easy to down.
import tensorflow as tf
import os
import sys
import tarfile
from six.moves import urllib

_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.
  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def main(_):
    if not tf.gfile.Exists('./flowers'):
        tf.gfile.MakeDirs('./flowers')

    download_and_uncompress_tarball(_DATA_URL, './flowers')
    os.rename('./flowers/flower_photos', "./flowers/images")

if __name__ == '__main__':
    tf.app.run()