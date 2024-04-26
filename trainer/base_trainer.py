import abc
import torch
import os


class BaseTrainer(abc.ABC):

    def load_device(self):
        torch.cuda.set_device(self.conf['device_id'])
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        return device


    def setup_train(self):
        """
        Loss, Optimizer, Scheduler
        """
        # self.loss_criterion = torch.nn.MSELoss()
        self.loss_criterion = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['learning_rate'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(
                        self.optimizer,
                        step_size=self.conf['decay_epoch'],
                        gamma=0.9
                    )
    

    def load_pretrained_model(self):

        ckpt_dir = './ckpt'
        ckpt_files = os.listdir(ckpt_dir)

        # Get selected model path
        model_path = None
        print("All pretrained model files as follows:")
        for i, file in enumerate(ckpt_files, 1):
            print(f"{i}. {file}")
        while selected_model := int(input("Choose the model to load (input the corresponding number): ")):
            if 0 < selected_model <= len(ckpt_files):
                model_path = os.path.join(ckpt_dir, ckpt_files[selected_model-1])
                print(f"Selected model: {ckpt_files[selected_model-1]}")
                break
            else:
                print("Invalid selection, please enter the correct number.")
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        print(f"model restored from {model_path}, start inference...")


    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError
    
    @abc.abstractmethod
    def validate_epoch(self, epoch):
        raise NotImplementedError
    
    @abc.abstractmethod
    def test_epoch(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError
