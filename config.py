import torch


class Config:
    MODEL_NAME = "MCAKT"
    INFO = "numworkers4 pin_memory bs128*2"
    DATASET = "AS12_lag3"
    Variants = ""
    # model_file = "/home/81076983/kr/MCAKT-V/lightning_logs/version_6/checkpoints/MCAKT-RF_epoch=49-step=3299.ckpt"
    DDP = True
    hp_conf = {
        "LR": 5e-4,  # E1e-3 A5e-5
        "DECAY": 0,
        "EMBED_DIMS": 128,
        "H_DIMS": 128,  # E64 A128
        "ENC_HEADS": 8,
        "NUM_ENCODER": 5,  # E5 A4
        "BATCH_SIZE": 128,  # E128,2 A64,1
        "MA_DROP_OUT": 0.5,
        "PL_DROP_OUT": 0,  # E0.1 A0
        "FFN_DROP_OUT": 0.5,
        "LAYER_NUM": 2,  # E3 A1
    }
    Forget = True
    # E True A False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 284
    DEVICE_NUM = [1]
    # LR = 5e-5
    # DECAY = 0
    CLIP = 0
    CON = 2
    MIN_SEQ = 5
    MAX_SEQ = 200
    EPOCH_NUM = 100

    # EMBED_DIMS = 128
    # ENC_HEADS = DEC_HEADS = 8
    # NUM_ENCODER = NUM_DECODER = 4
    # BATCH_SIZE = 64
    TRAIN_FILE = "../data/AS09/non_skill_builder_data_new.csv"
    QUESTION_FILE = "../data/AS09/ns_questions_pre.csv"
    # RAIEd2020: 13523  EdNetKT1: 18143(20000) ASs: 207349 ASns: 200958  AS12: 47205 AS12_3: 767144
    TOTAL_EXE = 767144
    # RAIEd2020: 200  EdNetKT1: 300(350) ASs: 401 ASns: 400 AS12: 243 AS12_3: 1642
    TOTAL_CAT = 1642
    MAX_CATS_PER = 10
    DIFF_NUM = 10
    # DROP_OUT = 0.5
    # LAYER_NUM = 1
    # 0.8 0.1 0.1: RA4 EN5 AS2 ASns2
    # (0.8 0.2) 0.1: RA5 EN6 AS ASns
    DATA_FILE_PATH = "../data/pre/AS12_lag3"
    TEST_FILE_PATH = "../data/pre/AS12_lag3"
    # RAIEd2020: 3  EdNetKT1: 2
    RESPONSES_NUM = 3
    RoPE = False
    Pos_Bias = True
    KQ_SAME = True
    # AS 5000 500 50
    # 301 1441 366
    LAG_S = 5000
    LAG_M = 500
    LAG_D = 50
