from trainer.ggbond_trainer import GGBondTrainer
from trainer.ggban_trainer import GGBanTrainer
from trainer.gaf_trainer import GAFTrainer
from argparse import ArgumentParser

parser = ArgumentParser(description='Just Run!')
parser.add_argument('--cfg_file', type=str, default='./config/PeMS08/config_server.json', help='Config File')
parser.add_argument('--model', type=str, default='gaf', help='Model Name')
parser.add_argument('--run_type', type=str, default='train', help='train or eval')
args = parser.parse_args()
if __name__ == "__main__":
    if args.model == 'gaf':
        dic_gaf={}
        dic_gaf['factor']=3
        dic_gaf['d_ff']=128
        dic_gaf['moving_avg']=13
        dic_gaf['encoder_layers']=2
        dic_gaf['decoder_layers']=1
        trainer = GAFTrainer(cfg_file=args.cfg_file, add_config_dict=dic_gaf)
    elif args.model == 'ggbond':
        trainer = GGBondTrainer(cfg_file=args.cfg_file)
    elif args.model == 'ggban':
        trainer = GGBanTrainer(cfg_file=args.cfg_file)
    
    if args.run_type == 'train':
        trainer.train()
    elif args.run_type == 'eval':
        trainer.eval()
