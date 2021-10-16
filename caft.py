from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
#from Weight import Weight
from config import seed, batch_size, epochs, lr, momentum, no_cuda, seed, log_interval, l2_decay, \
    class_num, param, bottle_neck, pseudo_label_th, window_size
import time
from FDA import FDA_source_to_target
import numpy as np
import random
#import torchvision
import wandb
import argparse

torch.manual_seed(seed)
if not no_cuda:
   torch.cuda.manual_seed(seed)
parser = argparse.ArgumentParser(description='DSAN for Unsupervised Domain Adaptation')
# dataset parameters
parser.add_argument('--root_path', type=str, default = "./dataset/")
parser.add_argument('--source_name', type=str, default="Art")
parser.add_argument('--target_name', type=str, default="Clipart" )
parser.add_argument("--wandb", type=int, default=1, help="log to wandb or not")
parser.add_argument("--cuda_id",type=str,default='1')
parser.add_argument("--cuda",type=bool, default=True)
parser.add_argument("--window_size",type=float,default=0.2)
args = parser.parse_args()
window_size = args.window_size
cuda_id = args.cuda_id
cuda = args.cuda
# wandb initialize
mode = 'online' if args.wandb else 'disabled'
wandb.init(project='CAFT', entity='vclab', name=f'{args.source_name} 2 {args.target_name} \
    window_size=: {args.window_size} for DSAN Office-home', mode=mode,tags=["DSAN"])
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

new_source_loader = data_loader.load_training(args.root_path, \
    args.source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(args.root_path, \
    args.target_name, 2*batch_size, kwargs)
target_test_loader = data_loader.load_testing(args.root_path, \
    args.target_name, batch_size, kwargs)
target_test_loader_no_shuffle = data_loader.load_testing_without_shuffle(args.root_path, \
    args.target_name, batch_size, kwargs)

len_source_dataset = len(new_source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_new_source_loader = len(new_source_loader)
len_target_loader = len(target_train_loader)
len_max_loader = np.maximum(len_new_source_loader,len_target_loader)


def train_with_fda(epoch, model):
    """
    This fucntion will train DA model with transformed source
    """
    running_cls_loss = 0.0
    running_loss = 0.0
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    pseudo_target_label = get_psuedo_target(model)
    print("Available distinct class for target for FDA =:{}".format(len(pseudo_target_label.keys())))
    model.train()

    iter_source = iter(new_source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_max_loader
    for i in range(1, num_iter):
        _data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        data_source_aug = torch.zeros_like(data_target)
        label_source_aug = torch.cat((label_source,label_source),0)
        target_to_transform = torch.zeros_like(_data_source)
        for index , label in enumerate(label_source):
            true_label = label.item()
            if(true_label in pseudo_target_label.keys()):
                temp = len(pseudo_target_label[true_label])
                rand_index = random.randrange(0,temp,1)
                target_to_transform[index] = pseudo_target_label[true_label][rand_index]
            else:
                if(len(pseudo_target_label.keys())):
                    rand_label = random.randrange(0,len(pseudo_target_label.keys()),1)
                    rand_label = list(pseudo_target_label.keys())[rand_label]
                    temp = len(pseudo_target_label[rand_label])
                    rand_index = random.randrange(0,temp,1)
                    target_to_transform[index] = pseudo_target_label[rand_label][rand_index]
                else:
                    target_to_transform = data_target[0:batch_size,:,:,:]
        data_source = FDA_source_to_target(_data_source, target_to_transform,L=window_size)
        data_source_aug[0:batch_size,:,:,:] = _data_source
        data_source_aug[batch_size:,:,:,:] = data_source
        
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if i % len_new_source_loader ==0:
            iter_source = iter(new_source_loader)
        if cuda:
            data_source_aug, label_source_aug = data_source_aug.cuda(), label_source_aug.cuda()
            data_target = data_target.cuda()
        data_source_aug, label_source_aug = Variable(data_source_aug), Variable(label_source_aug)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source_aug, data_target, label_source_aug)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source_aug)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + param * lambd * loss_mmd
        running_cls_loss+=loss_cls.item()
        running_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset,
                100. * i / len_new_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))
    wandb.log({"Cls Loss":running_cls_loss/num_iter,"Total Loss":running_loss/num_iter})
def get_psuedo_target(model):
    """
    This Function will return a dictionary with key as pseudo class label and value as samples w.r.t that class
    """
    model.eval()
    pseudo_target_label = {}
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, _ = model(data, data, target)
            pseudo_label,psedo_prob = torch.argmax(s_output,dim=1), torch.max(F.softmax(s_output,dim=1),dim=1)
            for index, label in enumerate(pseudo_label):
                label = label.item()
                #print(psedo_prob[0][index].item())
                if(psedo_prob[0][index].item() > pseudo_label_th):
                    if(label not in pseudo_target_label.keys()):
                        pseudo_target_label[label] = [data[index]]
                    else:
                        pseudo_target_label[label].append(data[index])
            
    return pseudo_target_label

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, _= model(data, data, target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            args.target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    model = models.DSAN(num_classes=class_num)
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    time_start=time.time()
    for epoch in range(1, epochs + 1):
        train_with_fda(epoch, model)
        t_correct = test(model)
        
        if t_correct > correct:
            correct = t_correct
            #torch.save(model, 'model.pkl')
        end_time = time.time()
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              args.source_name, args.target_name, correct, 100. * correct / len_target_dataset ))
        print('cost time:', end_time - time_start)
        wandb.log({"Target Accuracy":100. * t_correct / len_target_dataset, "Max Acc":100. * correct / len_target_dataset})
        if(epoch==100):
            model_path = "aug_dsan_with_fda_" + args.source_name + str(2) + args.target_name + str(epoch)+ "_.pth"
            torch.save(model,model_path)
    final_model_path = "final_aug_dsan_with_fda_" + args.source_name + str(2) + args.target_name + "_.pth"
    torch.save(model,final_model_path)