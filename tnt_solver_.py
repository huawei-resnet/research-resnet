import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchnet as tnt
from torchnet.engine import Engine
from dataset.data_loader_ import CIFAR10Data
from sklearn.base import BaseEstimator, ClassifierMixin
from models.resnet_ import resnet20, resnet56, resnet110
import torch.optim as optim
import torch.nn.functional as F
from sklearn.grid_search import GridSearchCV


# from torchnet.logger import VisdomLogger, VisdomPlotLogger


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


def main(model, opt, epoch, loss_fn=F.cross_entropy, lr_scheduler=None):
    """
    train model and test on test data
    :return:
    """
    num_classes = 10

    data = CIFAR10Data(train_split=0.8)
    train_itr = data.get_train_loader(batch_size=64)
    val_itr = data.get_val_loader(batch_size=64)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
    history = {'train_loss': [], 'train_acc': [], 'train_lr': [], 'val_loss': [], 'val_acc': []}

    tuned_params = {}
    gs = GridSearchCV(ResnetClassifier(), tuned_params)

    #     torch.manual_seed(6666)
#     torch.cuda.manual_seed(6666)
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')
#     model.to(device)
#
#     def reset_meters():
#         classacc.reset()
#         meter_loss.reset()
#
#     def h(sample):
#         x = sample[0].to(device)
#         y = sample[1].to(device)
#         o = model(x)
#         return loss_fn(o, y), o
#
#     def on_forward(state):
#         classacc.add(state['output'].detach(), state['sample'][1])
#         meter_loss.add(state['loss'].item())
#         confusion_meter.add(state['output'].detach(), state['sample'][1])
#         if state['train']:
#             state['iterator'].set_postfix_str(s="loss:{:.4f}, acc:{:.4f}%".format(meter_loss.value()[0], classacc.value()[0]))
#
#     def on_start_epoch(state):
#         current_lr = opt.param_groups[0]['lr']
#         print('Epoch: %d/%d, lr:%.2e' % (state['epoch']+1, state['maxepoch'], current_lr))
#         reset_meters()
#         model.train(True)
#         state['iterator'] = tqdm(state['iterator'], file=sys.stdout)
# #         lr_logger.log(state['epoch'], current_lr)
#         history['train_lr'].append(current_lr)
#
#     def on_end_epoch(state):
#         # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
# #         train_loss_logger.log(state['epoch'], meter_loss.value()[0])
# #         train_err_logger.log(state['epoch'], classacc.value()[0])
#         history['train_loss'].append(meter_loss.value()[0])
#         history['train_acc'].append(classacc.value()[0])
#
#         # do validation at the end of each epoch
#         reset_meters()
#         model.train(False)
#         engine.test(h, val_itr)
#         print('Val loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))
#
#         if lr_scheduler:
#             if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                 lr_scheduler.step(classacc.value()[0], epoch=(epoch+1))
#             else:
#                 lr_scheduler.step()
#
# #         test_loss_logger.log(state['epoch'], meter_loss.value()[0])
# #         test_err_logger.log(state['epoch'], classacc.value()[0])
# #         confusion_logger.log(confusion_meter.value())
#         history['val_loss'].append(meter_loss.value()[0])
#         history['val_acc'].append(classacc.value()[0])
#
#     engine = Engine()
#     engine.hooks['on_forward'] = on_forward
#     engine.hooks['on_start_epoch'] = on_start_epoch
#     engine.hooks['on_end_epoch'] = on_end_epoch
#     engine.train(h, train_itr, epoch, opt)
#
#     # test
#     test_itr = data.get_test_loader(batch_size=64)
#     model.train(False)
#     engine.test(h, test_itr)
#     print('Test loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))
    return history


class ResnetClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, model=resnet20(), data=CIFAR10Data(train_split=0.8), opt=optim.SGD(resnet20().parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False), epoches=2, loss_fn=F.cross_entropy, lr_scheduler=None):
        """
        Called when initializing the classifier
        """
        self.epoches = epoches
        self.loss_fn = loss_fn
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.data = data

        torch.manual_seed(6666)
        torch.cuda.manual_seed(6666)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.num_classes = 10

        self.meter_loss = tnt.meter.AverageValueMeter()
        self.classacc = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(self.num_classes, normalized=True)
        self.history = {'train_loss': [], 'train_acc': [], 'train_lr': [], 'val_loss': [], 'val_acc': []}


    def fit(self, data=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        self.train_itr = self.data.get_train_loader(batch_size=64)
        self.test_itr = self.data.get_test_loader(batch_size=64)
        self.val_itr = self.data.get_val_loader(batch_size=64)

        return self

    def predict(self, X=None, y=None):

        def h(sample):
            x = sample[0].to(self.device)
            y = sample[1].to(self.device)
            o = self.model(x)
            return self.loss_fn(o, y), o

        def reset_meters():
            self.classacc.reset()
            self.meter_loss.reset()

        def on_forward(state):
            self.classacc.add(state['output'].detach(), state['sample'][1])
            self.meter_loss.add(state['loss'].item())
            self.confusion_meter.add(state['output'].detach(), state['sample'][1])
            if state['train']:
                state['iterator'].set_postfix_str(
                    s="loss:{:.4f}, acc:{:.4f}%".format(self.meter_loss.value()[0], self.classacc.value()[0]))

        def on_start_epoch(state):
            current_lr = self.opt.param_groups[0]['lr']
            print('Epoch: %d/%d, lr:%.2e' % (state['epoch'] + 1, state['maxepoch'], current_lr))
            reset_meters()
            self.model.train(True)
            state['iterator'] = tqdm(state['iterator'], file=sys.stdout)
            #         lr_logger.log(state['epoch'], current_lr)
            history['train_lr'].append(current_lr)

        def on_end_epoch(state):
            # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
            #         train_loss_logger.log(state['epoch'], meter_loss.value()[0])
            #         train_err_logger.log(state['epoch'], classacc.value()[0])
            self.history['train_loss'].append(self.meter_loss.value()[0])
            self.history['train_acc'].append(self.classacc.value()[0])

            # do validation at the end of each epoch
            reset_meters()
            self.model.train(False)
            engine.test(h, self.val_itr)
            print('Val loss: %.4f, accuracy: %.2f%%' % (self.meter_loss.value()[0], self.classacc.value()[0]))

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.classacc.value()[0], epoch=(self.epoch + 1))
                else:
                    self.lr_scheduler.step()

            #         test_loss_logger.log(state['epoch'], meter_loss.value()[0])
            #         test_err_logger.log(state['epoch'], classacc.value()[0])
            #         confusion_logger.log(confusion_meter.value())
            self.history['val_loss'].append(self.meter_loss.value()[0])
            self.history['val_acc'].append(self.classacc.value()[0])

        engine = Engine()
        engine.hooks['on_forward'] = on_forward
        engine.hooks['on_start_epoch'] = on_start_epoch
        engine.hooks['on_end_epoch'] = on_end_epoch
        engine.train(h, self.train_itr, self.epoch, self.opt)

        # test
        self.model.train(False)
        engine.test(h, self.test_itr)
        print('Test loss: %.4f, accuracy: %.2f%%' % (self.meter_loss.value()[0], self.classacc.value()[0]))

        return self.classacc.value()[0]

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        return self.classacc.value()[0]
