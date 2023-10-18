from encoder_IRT_AutoInt import *

if __name__ == "__main__":
    start_time = time()
    if not Config.DDP:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        ddp_str = None
    else:
        ddp_str = "ddp"
    torch.manual_seed(Config.seed)  #为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)  #为所有GPU设置随机种子
    print("============" + Config.MODEL_NAME + "============")
    print("DATASET: ", Config.DATASET)
    print("INFO: ", Config.INFO)

    # search hp
    config = Config.hp_conf

    test_loader = get_test_dataloaders(bs=config["BATCH_SIZE"])
    model = Encoder_IRT_Modle(config)
    # model = load_checkpoint(model, "lightning_logs/version_230/checkpoints/epoch=2-step=2219.ckpt")
    trainer = pl.Trainer(gpus=Config.DEVICE_NUM, max_epochs=Config.EPOCH_NUM,
                         distributed_backend=ddp_str,
                         gradient_clip_val=Config.CLIP,
                         # plugins=DDPPlugin(find_unused_parameters=False),
                         )

    # test
    model = load_checkpoint(model, Config.model_file)
    trainer.test(model, test_dataloaders=test_loader)

    print("TOTAL TIME: ", time() - start_time)