# %%time
# import torch.optim as optim
# import torch.nn.functional as F
# from models.resnet_n import resnet_n




# to make cycle
import torch.optim as optim
import torch.nn.functional as F
from models.resnet_n import resnet_n
from tnt_solver_ import *

def auto_resnet(layer_j, class_i, lr_x, epoch_x, history, data_part=0.8, mini_batch = 64, conv_num = 2, skip_conn = False):
#     history = [] # ??
    model = resnet_n(layer_j, class_i, conv_num, skip_conn)
    print(model)
#     opt = optim.SGD(model.parameters(), lr = lr_x * 1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)
#     lr_scheduler= optim.lr_scheduler.MultiStepLR(opt, milestones=[91, 137], gamma=0.1) # ??
#     loss_fn = F.cross_entropy
#     history.append(main(model, class_i, opt, epoch_x, loss_fn=loss_fn, lr_scheduler=lr_scheduler, data_part=data_part, mini_batch=mini_batch)) # ??
    

    
def plt_different_history(history, legend):

    history_name = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_lr']
    colors = ['r', 'g', 'b', 'orange', 'purple', 'gray', 'cyan', 'magenta', '#4134FF', '#FF7B9E']

    for h_n in history_name:
        for num_i, history_i in enumerate(history, start=0):
            # for num_j, history_j in enumerate(history_i, start=0):
            plt.plot(history_i[h_n], colors[num_i])
        plt.xlabel('epoch')
        plt.ylabel(h_n)
        plt.legend(legend, loc='upper left')
        plt.show()


    # # train_loss for resnet 20, 56, 110 with cifar10/100
    # # plt.plot(history[0]['train_loss'], 'r', history[1]['train_loss'], 'g', history[2]['train_loss'], 'b', history[3]['train_loss'], 'orange', history[4]['train_loss'], 'purple', history[5]['train_loss'], 'gray')
    # plt.xlabel('epoch')
    # plt.ylabel('Loss value')
    # plt.legend(['train_r20_c10', 'train_r56_c10', 'train_r110_c10', 'train_r20_c100', 'train_r56_c100', 'train_r110_c100'], loc='upper left')
    # plt.show()
    #
    # # test_loss for resnet 20, 56, 110 with cifar10/100
    # plt.plot(history[0]['val_loss'], 'r', history[1]['val_loss'], 'g', history[2]['val_loss'], 'b', history[3]['val_loss'], 'orange', history[4]['val_loss'], 'purple', history[5]['train_loss'], 'gray')
    # plt.xlabel('epoch')
    # plt.ylabel('Loss value')
    # plt.legend(['test_r20_c10', 'test_r56_c10', 'test_r110_c10', 'test_r20_c100', 'test_r56_c100', 'test_r110_c100'], loc='upper left')
    # plt.show()
    #
    # # train_acc for resnet 20, 56, 110 with cifar10/100
    # plt.plot(history[0]['train_acc'], 'r', history[1]['train_acc'], 'g', history[2]['train_acc'], 'b', history[3]['train_acc'], 'orange', history[4]['train_acc'], 'purple', history[5]['train_acc'], 'gray')
    # plt.xlabel('epoch')
    # plt.ylabel('acc value')
    # plt.legend(['train_r20_c10', 'train_r56_c10', 'train_r110_c10', 'train_r20_c100', 'train_r56_c100', 'train_r110_c100'], loc='upper left')
    # plt.show()
    #
    # # test_acc for resnet 20, 56, 110 with cifar10/100
    # plt.plot(history[0]['val_acc'], 'r', history[1]['val_acc'], 'g', history[2]['val_acc'], 'b', history[3]['val_acc'], 'orange', history[4]['val_acc'], 'purple', history[5]['val_acc'], 'gray')
    # plt.xlabel('epoch')
    # plt.ylabel('acc value')
    # plt.legend(['test_r20_c10', 'test_r56_c10', 'test_r110_c10', 'test_r20_c100', 'test_r56_c100', 'test_r110_c100'], loc='upper left')
    # plt.show()
    #
    # # learning rate for resnet 20, 56, 110 with cifar10/100
    # plt.plot(history[0]['train_lr'], 'r', history[1]['train_lr'], 'g', history[2]['train_lr'], 'b', history[3]['train_lr'], 'orange', history[4]['train_lr'], 'purple', history[5]['train_lr'], 'gray')
    # plt.xlabel('epoch')
    # plt.ylabel('Train LR')
    # plt.legend(['train_r20_c10', 'train_r56_c10', 'train_r110_c10', 'train_r20_c100', 'train_r56_c100', 'train_r110_c100'], loc='upper left')
    # plt.show()