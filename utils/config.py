class DatasetConfig:

  def __init__(self, num_blocks, num_samples, sr, len_audio, num_audios_per_shard,
               num_classes, mean, std,
               num_train_audios, num_test_audios, num_val_audios, threshold=0.5):
    self.num_blocks = num_blocks
    self.num_samples = num_samples
    self.sr = sr
    self.len_audio = len_audio
    self.num_segments = len_audio * sr // num_samples  # 每个音频被切成10段，每段2.68s
    self.num_audios_per_shard = num_audios_per_shard

    self.num_train_audios = num_train_audios
    self.num_val_audios = num_val_audios
    self.num_test_audios = num_test_audios
    self.num_train_segs = num_train_audios * self.num_segments
    self.num_val_segs = num_val_audios * self.num_segments
    self.num_test_segs = num_test_audios * self.num_segments

    self.num_classes = num_classes
    self.threshold = threshold

    self.mean = mean
    self.std = std

# 修改成 18709 1825 5329
MTT_CONFIG = DatasetConfig(num_blocks=9, num_samples=59049, sr=22050, len_audio=29, num_audios_per_shard=100,
                           num_train_audios=18709, num_val_audios=1825, num_test_audios=5329,num_classes=50,
                           mean=-0.0001650025078561157, std=0.1551193743944168)
