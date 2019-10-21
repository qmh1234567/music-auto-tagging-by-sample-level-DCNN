""" 
@ author: Qmh
@ file_name: run.py
@ time: 2019:09:20:11:08
""" 

# here put the import lib
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import constants as c
import matplotlib.pyplot as plt
import argparse
import tqdm
from tensorflow.keras.models import load_model 
from models import Sample_CNN,proposed_CNN
from utils.batch import create_datasets
from utils.config import *
import math
import tensorflow.keras.backend as K
from progress.bar import Bar


# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)


def model_fn(input_shape):
    # model = Sample_CNN(input_shape,len(c.TAGS))
    model = proposed_CNN(input_shape,len(c.TAGS))
    print(model.summary())

    # 加载预训练模型
    # model.load_weights('./checkpoints/best_se.h5',by_name=True)

    num_params = int(sum([K.count_params(p) for p in set(model.trainable_weights)]))
    print(f"=>params:{num_params:,}")

    optimizer = tf.keras.optimizers.SGD(lr=0.01,momentum=0.9, nesterov=True, decay=1e-6)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def new_train_model(model_path):

    config = MTT_CONFIG

    tfrecord_path = os.path.join(c.DATA_DIR,'tfrecord')

    # Create training, validation, and test datasets.
    dataset_train, dataset_val, dataset_test = create_datasets(
        tfrecord_path, c.BATCH_SIZE,c.NUM_READERS, config)
    
    input_shape = (59049,1)
    model = model_fn(input_shape)

    ch_pt = ModelCheckpoint(model_path,monitor='val_loss',\
        save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    rd_pu = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2)

     # Train!
    steps_train = int(math.ceil(config.num_train_segs / c.BATCH_SIZE))

    steps_val = int(math.ceil(config.num_val_segs / c.BATCH_SIZE))

    model.fit(dataset_train, epochs=20, steps_per_epoch=steps_train,
                validation_data=dataset_val, validation_steps=steps_val,
                callbacks=[early_stopping, ch_pt,rd_pu])

    print('training done')

def new_evaluate_model(model_path):
    config = MTT_CONFIG
    tfrecord_path = c.SAVE_DIR

    # Create test datasets.
    dataset_test = create_datasets(
        tfrecord_path, c.BATCH_SIZE, c.NUM_READERS, config,only_test=True)
    model = load_model(model_path)

    evaluate(model,dataset_test,config,classes=c.TAGS)


def evaluate(model,dataset_test,config,classes=None):
    print("evaluate....")
    iterator = dataset_test.make_one_shot_iterator()
    seg,label = iterator.get_next()
    #(1 10 59049)
    seg_shape = tf.shape(seg)
    batch_size,num_segments,num_samples = seg_shape[0],seg_shape[1],seg_shape[2]
    num_classes = tf.shape(label)[1]
    seg = tf.reshape(seg,shape=(batch_size*num_segments,num_samples,1))
    pred_segs = model(seg)
    pred_segs = tf.reshape(pred_segs,shape=(batch_size,num_segments,num_classes))
    pred = tf.reduce_mean(pred_segs,axis=1)

    y_true,y_prob = [],[]
    # class_rocaucs = []
    sess = K.get_session()
    bar = Bar('predicting..',max=config.num_test_audios,fill='#', suffix='%(percent)d%%')
    while True:
        try:
            bar.next()
            label_batch,pred_batch = sess.run([label,pred],feed_dict={K.learning_phase():0})
            y_true.append(label_batch)
            y_prob.append(pred_batch)
        except tf.errors.OutOfRangeError:
            break
            bar.finish()
    y_true,y_prob = np.concatenate(y_true),np.concatenate(y_prob)

    rocauc = metrics.roc_auc_score(y_true, y_prob, average='macro')
    prauc = metrics.average_precision_score(y_true, y_prob, average='macro')

    y_pred = (y_prob > config.threshold).astype(np.float32)
    acc = metrics.accuracy_score(y_true, y_pred)

    f1 = metrics.f1_score(y_true, y_pred, average='samples')
    
    cls_roc_aucs = []
    cls_accs = [] 
    if classes is not None:
        print(f'\n=>individual scores of {len(classes)} classes')
        for i,c in enumerate(classes):
            cls_rocauc = metrics.roc_auc_score(y_true[:,i],y_prob[:,i])
            cls_acc = metrics.accuracy_score(y_true[:,i],y_pred[:,i])
            cls_roc_aucs.append(cls_rocauc)
            cls_accs.append(cls_acc)
            print(f'[{i:2} {c:30} rocauc={cls_rocauc:.4f} acc={cls_acc:.4f}]')
        print()
    np.save('./rocuac.npy',np.array(cls_roc_aucs))
    np.save('./accs.npy',np.array(cls_accs))

    print(f'=> Test scores: rocauc={rocauc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')



def predict_audios(model_path):
    y_true_lsit,y_pred_list = [],[]
    _,_,test_dataset = Create_Dataset()
    print(f"test samples={len(test_dataset[0])}")
    # load model
    model = load_model(model_path)
    paths,labels = test_dataset
    for i in tqdm.tqdm(range(len(paths))):
        try:
            x_test,y_true = load_each_data(paths,labels,i)
            y_pred = model.predict(x_test)
            y_true_lsit.append(y_true)
            y_pred_list.append(y_pred)
        except Exception as e:
            print(e)
            exit(-1)
    y_true = np.array(y_true_lsit)
    y_pred = np.squeeze(np.array(y_pred_list))
    return y_true,y_pred



def evaluate_model(y_true,y_pre,classes):
    roc_auc = metrics.roc_auc_score(y_true,y_pre)
    prauc = metrics.average_precision_score(y_true,y_pre,average='macro')
    y_pro = (y_pre > 0.5).astype(np.float32)
    acc = metrics.accuracy_score(y_true,y_pro)
    f1 = metrics.f1_score(y_true,y_pro,average='samples')
    # 每个标签的准确率和roc_auc
    class_accs = []
    cls_rocaucs = []
    if classes is not None:
        print(f'\n=> Individual scores of {len(classes)} classes')
        for i,c in enumerate(classes):
            cls_roc_auc = metrics.roc_auc_score(y_true[:,i],y_pre[:,i])
            cls_prauc = metrics.average_precision_score(y_true[:,i],y_pre[:,i])
            cls_acc = metrics.accuracy_score(y_true[:,i],y_pro[:,i])
            cls_f1 = metrics.f1_score(y_true[:,i],y_pro[:,i])
            print(f'[{i:2} {c:30}] roc_auc={cls_roc_auc:.4f} prauc={cls_prauc:.4f} \
                acc={cls_acc:.4f} f1={cls_f1:.4f}')
            print()
    print(f'=> Test scores: rocauc={roc_auc:.6f}\tprauc={prauc:.6f}\tacc={acc:.6f}\tf1={f1:.6f}')

def main(mode):
    if mode=='train':
        if not os.path.exists(c.CHECKPOINT_DIR):
            os.makedirs(c.CHECKPOINT_DIR)
        model_path = os.path.join(c.CHECKPOINT_DIR,'best.h5')
        # train_model(model_path)
        new_train_model(model_path)
    else:
        model_path = os.path.join(c.CHECKPOINT_DIR,'best.h5')
        # y_true,y_pred = predict_audios(model_path)
        # evaluate_model(y_true,y_pred,c.TAGS)
        new_evaluate_model(model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['train','test']:
        print("you need add extra prama, train or test")
        exit()
    mode = sys.argv[1]
    main(str(mode))
