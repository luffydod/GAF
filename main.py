# from trainer.ggbond_trainer import GGBondTrainer
# from trainer.ggban_trainer import GGBanTrainer
from trainer.gaf_trainer import GAFTrainer
if __name__ == "__main__":
    trainer = GAFTrainer(cfg_file='./config/PeMS08/mini_config.json')
    # trainer = GGBondTrainer(cfg_file='./config/PeMS08/mini_config.json')
    trainer.train()
    # trainer.eval()
