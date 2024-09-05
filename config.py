import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# basic informations for data
NUM_CLASSES = 2
IMAGE_SIZE = 256
IMAGE_CHANNEL = 3

# paths for dataset
save_model_dir = "saved_model/model"
image_path = r'data\images\1. Training'
label_path = r'data\labels\1. Training.csv'

# configurations for train
EPOCHS = 25
BATCH_SIZE = 32
k_folds = 5
trainer_config = {
    "max_epochs": 20,
    "accelerator": "gpu",  # CPU 사용 시 "cpu"
    "precision": 16,  # Mixed precision
    "logger": TensorBoardLogger("logs/"),  # 로그 저장
    "callbacks": [ModelCheckpoint(monitor="val_loss"), EarlyStopping(monitor="val_loss", patience=3)],
    "log_every_n_steps": 50,
    "check_val_every_n_epoch": 1,
    "profiler": "simple",  # 학습 속도 분석
    "fast_dev_run": False,  # 코드 디버깅
    "sync_batchnorm": True,  # 다중 GPU 배치 정규화 동기화
    "deterministic": True,  # 결과 재현성 보장
}
