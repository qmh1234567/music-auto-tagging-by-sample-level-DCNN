""" 
@ author: Qmh
@ file_name: audio.py
@ time: 2019:09:25:10:25
""" 
import librosa
import tensorflow as tf
import numpy as np


def to_tfrecord_examples(row,config,sequence):
    audio_path,label = row['path'],row['label']
    sr,num_samples,num_segments,len_audio = config.sr,config.num_samples,config.num_segments,config.len_audio
    audio = load_audio(audio_path,sr,len_audio)
    # 将音频分段
    segments = [audio[i*num_samples:(i+1)*num_samples] for  i in range(num_segments)]
    if sequence:
        examples = [segment_to_sequence_example(segments,label)]
    else:
        examples = [segment_to_example(segment,label) for segment in segments]
    return examples


# 返回每个音频片段的样例
def segment_to_example(segment,label):
    raw_segment = np.array(segment,dtype=np.float32).reshape(-1).tostring()
    raw_label = np.array(label,dtype=np.uint8).reshape(-1).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label':bytes_feature(raw_label),
        'segment':bytes_feature(raw_segment)
    }))
    return example

# 返回单个完整音频的样例
def segment_to_sequence_example(segments,label):
    # 将每段音频数据转化成字符串
    raw_segments = [np.array(segment,dtype=np.float32).reshape(-1).tostring() for segment in segments]
    raw_label = np.array(label,dtype=np.uint8).reshape(-1).tostring()

    sequence_example = tf.train.SequenceExample(
        # 非序列化部分
        context = tf.train.Features(feature={
            'label':bytes_feature(raw_label)
        }),
        # 可变长序列
        feature_lists = tf.train.FeatureLists(feature_list={
            'segments': bytes_feature_list(raw_segments)
        })
    )
    return sequence_example

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_feature_list(values):
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def load_audio(path,sr,len_audio):
    audio,_ = librosa.load(path,sr=sr,duration=len_audio)
    total_samples = sr * len_audio
    if len(audio) < total_samples:
        audio = np.repeat(audio,total_samples//len(audio)+1)[:total_samples]
    return audio
