""" 
@ author: Qmh
@ file_name: constants.py
@ time: 2019:09:20:09:02
""" 
import os


DATA_DIR = "/home/dsp/Documents/wav_data/music_classification_dataset/MagnaTagATune/mtt/raw"

ANNOTA_PATH = os.path.join(DATA_DIR, "annotations_final.csv")  # annotations file
MP3_DIR = os.path.join(DATA_DIR,'mp3')
SAVE_DIR = os.path.join(DATA_DIR,'tfrecord')
CHECKPOINT_DIR = "./checkpoints"


DURA = 29
SR = 22050
NUM_SAMPLES = 59049
NUM_SEGMENTS = 10

# train
BATCH_SIZE = 23
NUM_READERS = 8


# TRAIN and TEST
TAGS = ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
        'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
        'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
        'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
        'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
        'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
        'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
        'slow', 'classical', 'guitar']