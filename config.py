import torch


class Config:
    VISUAL = False
    MODEL_NAME = "ACAKT-A"
    INFO = "ACAKT_A 2FFN Pos_Bias"
    DATASET = "AS12_lag3"
    AT = "FGM"
    NEG = False
    MCAKT = True
    AutoInt = True
    EXEB = True
    LOSS = 0
    samp = 0.01
    beta = 0.2
    theta = 0.1
    temp = 0.02
    epsilon = 10
    # model_file = "/home/81076983/kr/MCAKT-V/lightning_logs/version_6/checkpoints/MCAKT-RF_epoch=49-step=3299.ckpt"
    DDP = True
    hp_conf = {
        "LR": 5e-4,  # E1e-3 A5e-5
        "DECAY": 0,
        "EMBED_DIMS": 128,
        "H_DIMS": 64,  # E64 A128
        "ENC_HEADS": 8,
        "NUM_ENCODER": 5,  # E5 A4
        "BATCH_SIZE": 128,  # E128,2 A64,1
        "MA_DROP_OUT": 0.1,
        "PL_DROP_OUT": 0,  # E0.1 A0
        "FFN_DROP_OUT": 0.1,
        "LAYER_NUM": 2,
    }
    Forget = True  # E True A False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 284
    DEVICE_NUM = []
    CLIP = 0
    CON = 2
    MIN_SEQ = 5
    MAX_SEQ = 200
    EPOCH_NUM = 150
    TRAIN_FILE = "../data/AS12/AS12.csv"
    # QUESTION_FILE = "../data/AS12/AS12_problem_pre.csv"
    # RAIEd2020: 13523  EdNetKT1: 18143(20000) ASs: 207349 ASns: 200958 AS12: 47205
    TOTAL_EXE = 767144
    # RAIEd2020: 200  EdNetKT1: 300(350) ASs: 401 ASns: 400 AS12: 243
    TOTAL_CAT = 1642
    MAX_CATS_PER = 10
    DIFF_NUM = 10
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
    LAG_S = 5001
    LAG_M = 501
    LAG_D = 51
