from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, X, Y, TE, device='cpu'):
        self.X = X
        self.Y = Y
        self.TE = TE
        self.device = device

    def __len__(self):
        # num_samples
        return self.Y.shape[0]
    
    def __getitem__(self, index):
        X = self.X[index].clone().detach().to(self.device)
        Y = self.Y[index].clone().detach().to(self.device)
        TE = self.TE[index].clone().detach().to(self.device)
        return (X, Y, TE)