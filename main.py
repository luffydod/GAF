import torch
from model.GGBond import GGBond
from train import train
from utils.utils import load_config, load_data, plot_train_val_loss
from utils.utils import count_parameters

# set GPU number
torch.cuda.set_device(7)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conf = load_config()
data = load_data(conf)
data = {key: value.to(device) for key, value in data.items()}

model = GGBond(
            data['SE'],
            conf['num_his'],
            conf['num_heads'],
            conf['dim_heads'],
            conf['num_block'],
            conf['bn_momentum']
        ).to(device)

loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=conf['learing_rate'])

# optimizer to gpu
optimizer_state_dict = optimizer.state_dict()
for key, value in optimizer_state_dict.items():
    if isinstance(value, torch.Tensor):
        optimizer_state_dict[key] = value.to(device)
optimizer.load_state_dict(optimizer_state_dict)

scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=conf['decay_epoch'],
                gamma=0.9
            )
parameters = count_parameters(model)
print('trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    
    train_total_loss, val_total_loss = train(model, conf, data, loss_criterion, optimizer, scheduler)
    plot_train_val_loss(train_total_loss, val_total_loss)