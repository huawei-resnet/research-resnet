import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchnet as tnt
from torchnet.engine import Engine
from dataset.data_loader_ import CIFAR10Data
# from torchnet.logger import VisdomLogger, VisdomPlotLogger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from models.resnet_ import resnet20, resnet56, resnet110
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
import torch.optim as optim
from config.config_ import mean, std
import torchvision.transforms as transforms

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
    
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.to(device)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_set = torchvision.datasets.cifar.CIFAR10('data/', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=val_transform)

    train_split=0.9
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(num_train * train_split)
    train_idx, val_idx = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(train_set, train_idx)
    val_dataset = torch.utils.data.Subset(val_set, val_idx)

    X = (np.swapaxes(train_set.train_data, 1, 3)/255).astype('float32')
    y = np.array(val_set.train_labels).astype('int64')    
    
#     net1 = ResnetClassifier(1)
#     net1.fit(train_set,val_set)
    
    net = ResnetClassifier()
    
    tuned_params = {"epoches" : [5,10,15]}

    gs = GridSearchCV(estimator=net, param_grid=tuned_params, refit=False, cv=3, scoring='accuracy')
#     gs.fit(train_set, val_set)
    gs.fit(X, y)
#     gs.fit(train_dataset, val_dataset)
    best_score = gs.best_score_
    best_params = gs.best_params_
    
class ResnetClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, epoches=0):
        """
        Called when initializing the classifier
        """
        self.epoches = epoches

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
#         train_split=0.9

#         num_train = len(X)
#         indices = list(range(num_train))
#         split = int(num_train * train_split)
#         train_idx, val_idx = indices[:split], indices[split:]
#         train_dataset = torch.utils.data.Subset(X, train_idx)
#         val_dataset = torch.utils.data.Subset(y, val_idx)
#         test_dataset = val_dataset

        print(type(X))
        print(type(y))
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y)
        print(type(tensor_x))
        print(type(tensor_y))

        train_loader = torch.utils.data.DataLoader(
            tensor_x, batch_size=128,
            num_workers=2, shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(
            tensor_y, batch_size=128,
            num_workers=2, shuffle=False
        )
            
        test_loader = torch.utils.data.DataLoader(
            tensor_y, batch_size=64,
            num_workers=2, shuffle=False
        )
              
        print(train_loader.dataset.size())
        print(val_loader.dataset.size())
        
        num_classes = 10
        lr_scheduler = None
        
        model = resnet20()

        loss_fn = F.cross_entropy

        torch.manual_seed(6666)
        torch.cuda.manual_seed(6666)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        model.to(device)
        history = {'train_loss': [], 'train_acc': [], 'train_lr': [], 'val_loss': [], 'val_acc': []}
        
        opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)

        meter_loss = tnt.meter.AverageValueMeter()
        classacc = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)

        def reset_meters():
            classacc.reset()
            meter_loss.reset()

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
            history['train_lr'].append(current_lr)

        def on_end_epoch(state):
            history['train_loss'].append(meter_loss.value()[0])
            history['train_acc'].append(classacc.value()[0])

            # do validation at the end of each epoch
            reset_meters()
            model.train(False)
            engine.test(h, val_loader)
            print('Val loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))

            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(classacc.value()[0], epoch=(epoch+1))
                else:
                    lr_scheduler.step()

            history['val_loss'].append(meter_loss.value()[0])
            history['val_acc'].append(classacc.value()[0])
            
        def h(sample):
            x = sample[0].to(device)
            print(type(sample))
            print(sample)
            print(type(x))
            print(x)
            y = sample[1].to(device)
            o = model(x.cuda())
            return loss_fn(o, y), o

        engine = Engine()
        engine.hooks['on_forward'] = on_forward
        engine.hooks['on_start_epoch'] = on_start_epoch
        engine.hooks['on_end_epoch'] = on_end_epoch
        engine.train(h, train_loader, self.epoches, opt)

        # test
        model.train(False)
        engine.test(h, test_loader)
        print('Test loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classacc.value()[0]))
        plot_history(history)
        return self

#     def predict(self, X=None, y=None):
#         return self.classacc.value()[0]

#     def score(self, X=None, y=None):
#         # counts number of values bigger than mean
#         return self.classacc.value()[0]

