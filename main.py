from trainer.ggbond_trainer import GGBondTrainer


if __name__ == "__main__":
    trainer = GGBondTrainer(cfg_file='./config/mini_config.json')
    trainer.train()
    # trainer.eval()
