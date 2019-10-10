""" 
@ author: Qmh
@ file_name: build_dataset.py
@ time: 2019:09:25:09:15
""" 

import pandas as pd
import numpy as np
from multiprocessing import Pool
import constants as c
import os 
import tensorflow as tf
from sklearn.utils import shuffle
from audio import to_tfrecord_examples
from config import *
import tqdm

def split_by_directory(mp3_path):
    directory = mp3_path.split('/')[0]
    part = int(directory,16)
    if part in range(12):
        return 'train'
    elif part is 12:
        return 'val'
    elif part in range(13,16):
        return 'test'

# 划分数据集，对数据进行分组
def make_dataset_info(df,num_audios_per_shard=100):
    # train:18709 1825 5329
    df['split'] = df['mp3_path'].apply(lambda mp3_path:split_by_directory(mp3_path))
    df = shuffle(df)
    df = df.copy()
    for split in ['train','val','test']:
        num_audios = sum(df['split'] == split)
        print(f"=>{split} num_audios={num_audios}")
        num_shards = num_audios // num_audios_per_shard
        num_remainders = num_audios % num_audios_per_shard  # 9
        # title(A,n)将某个数组重复n次 187-> 18700
        shards = np.tile(np.arange(num_shards),num_audios_per_shard)
        # shape = (18709,)
        shards = np.concatenate([shards,np.arange(num_remainders) % num_shards])
        # 返回一个乱序数组
        shards = np.random.permutation(shards)
        # 增加 split shard列
        df.loc[df['split']==split,'shard'] = shards

    df['shard'] = df['shard'].astype(int)
    paths = list(anno_df['mp3_path'].values)
    paths = [os.path.join(c.MP3_DIR,path) for path in paths]
    labels = list(anno_df[c.TAGS].values)
    df = pd.DataFrame({'id': df['clip_id'],'label':labels,'split':df['split'],'shard':df['shard'],'path':paths})
    return df 


# Create a pool for multi-processing.
# The number of processes will be set as same as the number of cpus.
def new_preprocess(df,config):
    if not os.path.exists(c.SAVE_DIR):
        os.makedirs(c.SAVE_DIR)
    with Pool(processes=None) as pool:
        for split in ['train','val','test']:
            print(f'=>processing {split}')
            df_split = df[df['split']==split]
            shards = df_split.shard.unique()# 187
            for shard in tqdm.tqdm(sorted(shards)):
                df_split_shard = df_split[df_split['shard']==shard]
                # 设置文件名 187
                filename = f'{split}-{shard+1:04d}-{len(shards):04d}.tfrecord'
                filepath = os.path.join(c.SAVE_DIR,filename)
                if os.path.exists(filepath):
                    continue
                #  # 写入TFRecord文件 对输入数据做统一管理的格式
                with tf.io.TFRecordWriter(filepath) as writer:
                    list_args = []
                    # DataFrame迭代为(insex, Series)对, Series 包含 label,split,shard,path
                    for _,row in df_split_shard.iterrows():
                        list_args.append((row,config,split))
                    for i,examples in enumerate(pool.imap(process_audio,list_args)):
                        for example in examples:
                            writer.write(example.SerializeToString())
                        

def process_audio(args):
    row,config,split = args
    try:
        sequence = True if split == 'test' else False
        examples = to_tfrecord_examples(row,config,sequence)
    except Exception as e:
        print('=> Error: cannot load audio (reason below): ' + row['path'])
        print(e)
        examples = []
    return examples



if __name__ == "__main__":
    anno_df = pd.read_csv(c.ANNOTA_PATH,delimiter='\t')
    df = make_dataset_info(anno_df)
    config = MTT_CONFIG
    new_preprocess(df,config)
    