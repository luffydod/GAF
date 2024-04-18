from trainer.ggbond_trainer import GGBondTrainer


if __name__ == "__main__":
    trainer = GGBondTrainer(cfg_file='./config/PeMS04/mini_config.json')
    trainer.train()
    # trainer.eval()
