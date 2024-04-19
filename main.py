from trainer.ggbond_trainer import GGBondTrainer
from trainer.ggban_trainer import GGBanTrainer

if __name__ == "__main__":
    trainer = GGBanTrainer(cfg_file='./config/PeMS08/mini_config.json')
    # trainer = GGBondTrainer(cfg_file='./config/PeMS08/mini_config.json')
    # trainer.train()
    trainer.eval()
