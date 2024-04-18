from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, X, Y, TE):
        self.X = X
        self.Y = Y
        self.TE = TE

    def __len__(self):
        # num_samples
        return self.Y.shape[0]
    
    def __getitem__(self, index):
        X = self.X[index].clone().detach()
        Y = self.Y[index].clone().detach()
        TE = self.TE[index].clone().detach()
        return (X, Y, TE)