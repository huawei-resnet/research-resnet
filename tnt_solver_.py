import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchnet as tnt
from torchnet.engine import Engine
from dataset.data_loader_ import CIFAR10Data, CIFAR100Data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history):
    """
    plot loss and acc history.
    :param history: train returned history object
    """
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history['train_lr'])
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.show()

# add used part of data
def main(model, num_classes, opt, epoch, loss_fn=F.cross_entropy, lr_scheduler=None, data_part=0.8, mini_batch = 64): # trying to override function
    """
    train model and test on test data
    :return:
    """
    # num_classes = 10

    if num_classes == 10:
        data = CIFAR10Data(train_split=data_part)#0.8)
    else:
        data = CIFAR100Data(train_split=data_part)#0.8)

    train_itr = data.get_train_loader(batch_size=mini_batch)#64)
    val_itr = data.get_val_loader(batch_size=mini_batch)#64)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
    history = {'train_loss': [], 'train_acc': [], 'train_lr': [], 'val_loss': [], 'val_acc': []}

    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.to(device)

    def reset_meters():
        classacc.reset()
        meter_loss.reset()

    def h(sample):
        x = sample[0].to(device)
        y = sample[1].to(device)
        o = model(x)
        return loss_fn(o, y), o

    def on_forward(state):
        classacc.add(state['output'].detach(), state['sample'][1])
        meter_loss.add(state['loss'].item())
        confusion_meter.add(state['output'].detach(), state['sample'][1])
        if state['train']:
            state['iterator'].set_postfix_str(s="loss:{:.4f}, acc:{:.4f}%".format(meter_loss.value()[0], classacc.value()[0]))

    def on_start_epoch(state):
        current_lr = opt.param_groups[0]['lr']
        print('Epoch: %d/%d, lr:%.2e' % (state['epoch']+1, state['maxepoch'], current_lr))
        reset_meters()
        model.train(True)
        state['iterator'] = tqdm(state['iterator'], file=sys.stdout)
#         lr_logger.log(state['epoch'], current_lr)
        history['train_lr'].append(current_lr)

    def on_end_epoch(state):
        # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
#         train_loss_logger.log(state['epoch'], meter_loss.value()[0])
#         train_err_logger.log(state['epoch'], classacc.value()[0])
        history['train_loss'].append(meter_loss.value()[0])
        history['train_acc'].append(classacc.value()[0])

        # do validation at the end of each epoch
        reset_meters()
        model.train(False)
        engine.test(h, val_itr)
        print('Val loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))

        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(classacc.value()[0], epoch=(epoch+1))
            else:
                lr_scheduler.step()

#         test_loss_logger.log(state['epoch'], meter_loss.value()[0])
#         test_err_logger.log(state['epoch'], classacc.value()[0])
#         confusion_logger.log(confusion_meter.value())
        history['val_loss'].append(meter_loss.value()[0])
        history['val_acc'].append(classacc.value()[0])

    engine = Engine()
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, train_itr, epoch, opt)

    # test
    test_itr = data.get_test_loader(batch_size=64)
    model.train(False)
    engine.test(h, test_itr)
    print('Test loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))
    return history
