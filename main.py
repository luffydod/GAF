import torch
from model.GGBond import GGBond
from train import train
from utils.utils import load_config, load_data, plot_train_val_loss
from utils.utils import count_parameters

conf = load_config()
data = load_data(conf)

# set GPU number
torch.cuda.set_device(conf['device_id'])
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # load data To GPU
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
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=conf['decay_epoch'],
                    gamma=0.9
                )
    # 打印模型中可训练参数的数量
    count_parameters(model)
    
    train_total_loss, val_total_loss = train(model, conf, data, loss_criterion, optimizer, scheduler)
    # To CPU
    train_total_loss = [loss.cpu().detach().numpy() for loss in train_total_loss]
    val_total_loss = [loss.cpu().detach().numpy() for loss in val_total_loss]
    # Plot Loss Curve
    plot_train_val_loss(train_total_loss, val_total_loss)