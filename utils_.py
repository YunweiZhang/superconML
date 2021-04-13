import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
import pandas as pd

def validate(test_set, model):

    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(test_set['data']).float()
        targets = torch.from_numpy(test_set['targets']).long()
        outputs = model(inputs)

        prec = accuracy(outputs, targets, topk=(1,))
        prob = torch.softmax(outputs, dim=1).data.numpy()

    return  prec, prob

def prob(unlabeled_set, model):

    model.eval()
    num_u = np.shape(unlabeled_set['data'])[0]
    prob_list = []
    with torch.no_grad():

        for i in range(num_u):

            inputs_u = torch.from_numpy(unlabeled_set['data'][i:i+1,:]).float()

            outputs_u = model(inputs_u)

            p = torch.softmax(outputs_u, dim=1).data.numpy()

            prob_list.append(p)

    return  prob_list



class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)


    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def get_conductor(random_state):

    from sklearn.model_selection import train_test_split
    input_u_1 = np.genfromtxt('dataset/data_test_mp_binary.txt', delimiter="   ",  usecols=[2, 3, 4, 5, 6, 7, 8])
    input_u_2 = np.genfromtxt('dataset/data_test_mp_ternary.txt', delimiter="   ", usecols=[2, 3, 4, 5, 6, 7, 8])
    input_l = np.genfromtxt('dataset/train_data.txt', delimiter="   ",  usecols=[2, 3, 4, 5, 6, 7, 8, 9])


    train_labeled_dataset, train_unlabeled_dataset_1, train_unlabeled_dataset_2,  train_unlabeled_dataset_3, test_dataset = {}, {}, {}, {}, {}

    input_1 = []
    in_list = []
    for i in range(np.shape(input_u_1)[0]):
        sd = np.isinf(input_u_1[i, :]).any() * 1.0
        if sd == 0:
            input_1.append(input_u_1[i, :])
            in_list.append(i)

    line_index = 0
    string_index_1 = []
    with open('dataset/data_test_mp_binary.txt', "r") as file:
        for line in file:
            if line_index in in_list:
                a = line.split()
                string_index_1.append(a[-2:])

            line_index +=1

    input_2 = []
    in_list = []
    for i in range(np.shape(input_u_2)[0]):
        sd = np.isinf(input_u_2[i, :]).any() * 1.0
        if sd == 0:
            input_2.append(input_u_2[i, :])
            in_list.append(i)

    line_index = 0
    string_index_2 = []
    with open('dataset/data_test_mp_ternary.txt', "r") as file:
        for line in file:
            if line_index in in_list:
                a = line.split()
                string_index_2.append(a[-2:])

            line_index +=1

    X_train, X_test, y_train, y_test = train_test_split(input_l[:,:-1], input_l[:, -1], test_size=0.01, random_state=random_state)

    train_labeled_dataset['data'] = X_train
    train_labeled_dataset['targets'] = np.expand_dims(y_train,axis=1)

    train_unlabeled_dataset_1['data'] = np.array(input_1)
    train_unlabeled_dataset_1['targets'] =  np.expand_dims([-1]* np.shape(input_1)[0],axis=1)

    train_unlabeled_dataset_2['data'] = np.array(input_2)
    train_unlabeled_dataset_2['targets'] =  np.expand_dims([-1]*np.shape(input_2)[0],axis=1)

    test_dataset['data'] = X_test
    test_dataset['targets'] = np.expand_dims(y_test,axis=1)


    return train_labeled_dataset, train_unlabeled_dataset_1, train_unlabeled_dataset_2, test_dataset, string_index_1, string_index_2






class ConductorNet(nn.Module):
    def __init__(self):
        super(ConductorNet, self).__init__()

        # 1st block
        self.block1 = nn.Linear(7,16)
        # 2nd block
        self.block2 = nn.Linear(16,16)

        self.block3 = nn.Linear(16,16)
        # 3rd block

        self.fc = nn.Linear(16, 2)


    def forward(self, x):

        out = F.relu(self.block1(x))
        out = F.relu(self.block2(out))
        out = F.relu(self.block3(out))

        return self.fc(out)



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def result_out(f_list,f_final_list,name_list):
    ff = np.round(np.mean(f_list, axis=0),4)

    f_final_list.append(ff)

    final_list = np.mean(f_final_list,axis=0)
    std_list = np.std(f_final_list, axis=0)
    sort_index = np.argsort(final_list)[::-1]

    df1 = pd.DataFrame([name_list[i] for i in sort_index])
    df2 = pd.DataFrame(final_list[sort_index])
    df3 = pd.DataFrame(std_list[sort_index])
    frames = [df1, df2, df3]
    result = pd.concat(frames, axis=1)

    return result, f_final_list