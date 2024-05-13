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
    
    @abc.abstractmethod
    def load_pretrained_model(self):
        raise NotImplementedError


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
