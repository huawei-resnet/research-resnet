import torch.optim as optim
import torch.nn.functional as F
from models.resnet_n import resnet_n
from tnt_solver_ import *

def auto_resnet(layer_j, class_i, lr_x, epoch_x, history, data_part=0.8, mini_batch = 64, conv_num = 2, skip_conn = False):
    model = resnet_n(layer_j, class_i, conv_num, skip_conn)
    print(model)
    opt = optim.SGD(model.parameters(), lr = lr_x * 1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler= optim.lr_scheduler.MultiStepLR(opt, milestones=[91, 137], gamma=0.1)
    loss_fn = F.cross_entropy
    history.append(main(model, class_i, opt, epoch_x, loss_fn=loss_fn, lr_scheduler=lr_scheduler, data_part=data_part, mini_batch=mini_batch))

# TODO: Should be removed due to duplication with def auto_resnet(...)
# Found this issue too late to solve with accuracy for all call locations
def auto_resnet_opt_task1_epochs_multiplier(layer_j, class_i, lr_x, epoch_x, history, milestones, data_part=0.8, mini_batch = 64, conv_num = 2, skip_conn = False):
    model = resnet_n(layer_j, class_i, conv_num, skip_conn)
    print(model)
    opt = optim.SGD(model.parameters(), lr = lr_x * 1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler= optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    loss_fn = F.cross_entropy
    history.append(main(model, class_i, opt, epoch_x, loss_fn=loss_fn, lr_scheduler=lr_scheduler, data_part=data_part, mini_batch=mini_batch))
    
def plt_different_history(history, legend):
    history_name = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_lr']
    colors = ['r', 'g', 'b', 'orange', 'purple', 'gray', 'cyan', 'magenta', '#4134FF', '#FF7B9E']

    for h_n in history_name:
        for num_i, history_i in enumerate(history, start=0):
            plt.plot(history_i[h_n], colors[num_i])
        plt.xlabel('epoch')
        plt.ylabel(h_n)
        plt.legend(legend, loc='upper left')
        plt.show()
