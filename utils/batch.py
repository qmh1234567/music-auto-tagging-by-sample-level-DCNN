import tensorflow as tf
from glob import glob


def tfrecord_parser(config):
  def parse_fn(example):
    # 解析单个张量
    features = tf.parse_single_example(example, features={
      'label': tf.FixedLenFeature([], tf.string),
      'segment': tf.FixedLenFeature([], tf.string)
    })
    # 将原始数据从字符串中取出
    segment = tf.decode_raw(features['segment'], tf.float32)
    segment = (segment - config.mean) / config.std  # standardization
    segment = tf.expand_dims(segment, axis=-1)

    label = tf.decode_raw(features['label'], tf.uint8)
    # 数据类型转换
    label = tf.cast(label, tf.float32)

    return segment, label

  return parse_fn


def tfrecord_parser_sequence(config):
  def parse_fn(sequence_example):
    # 创建解析器解析数据 标签和数据格式
    # http://www.pianshen.com/article/719643562/
    context, sequence = tf.parse_single_sequence_example(
      sequence_example,
      context_features={
        #  固定长度
        'label': tf.FixedLenFeature([], tf.string)
      },
      sequence_features={
        # 可变长
        'segments': tf.FixedLenSequenceFeature([], tf.string)
      })
    # 将字符串的字节重新解释为数字向量。
    segments = tf.decode_raw(sequence['segments'], tf.float32)

    segments = (segments - config.mean) / config.std  # standardization
    # -1代表最后一维
    segments = tf.expand_dims(segments, axis=-1)

    label = tf.decode_raw(context['label'], tf.uint8)
    # 数据类型转化
    label = tf.cast(label, tf.float32)

    return segments, label

  return parse_fn


def create_datasets(tfrecord_path, batch_size, num_readers, config, only_test=False):
  batch_size_test = max(1, batch_size // config.num_segments)
  #  glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
  filenames_test = glob(tfrecord_path + '/test-*.tfrecord')
  # 解析tfrecord文件的每条记录
  dataset_test = tf.data.TFRecordDataset(filenames_test)
  #  num_parallel_calls 处理数据的并发数
  dataset_test = dataset_test.map(tfrecord_parser_sequence(config), num_parallel_calls=num_readers)
  dataset_test = dataset_test.batch(batch_size_test)
  dataset_test = dataset_test.prefetch(2 * batch_size_test)

  if only_test:
    return dataset_test
  else:
    filenames_train = glob(tfrecord_path + '/train-*.tfrecord')
    dataset_train = tf.data.TFRecordDataset(filenames_train)
    dataset_train = dataset_train.map(tfrecord_parser(config), num_parallel_calls=num_readers)
    dataset_train = dataset_train.shuffle(buffer_size=10000)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.prefetch(2 * batch_size)

    filenames_val = glob(tfrecord_path + '/val-*.tfrecord')
    dataset_val = tf.data.TFRecordDataset(filenames_val)
    dataset_val = dataset_val.map(tfrecord_parser(config), num_parallel_calls=num_readers)
    # NOTE: Do not shuffle validation set.
    dataset_val = dataset_val.batch(batch_size)
    dataset_val = dataset_val.repeat()
    dataset_val = dataset_val.prefetch(2 * batch_size)

    return dataset_train, dataset_val, dataset_test
