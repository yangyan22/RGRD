import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from net.net import net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from utils import *
import random
from net.losses import L_loss, R_loss, rec_loss, kl_loss
from torchvision import transforms
import torchvision.transforms as transforms
from measure import metrics


# Training settings
parser = argparse.ArgumentParser(description='Pytorch RGRD')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='2000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123456789')
parser.add_argument('--data_train', type=str, default='train-dataset/LOL_SICE_LOLv2')
parser.add_argument('--data_test', type=str, default='test-dataset/LOL/raw')
parser.add_argument('--perceptual_test', type=str, default='./p_amber')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--weights', default='weights/', help='Location to save checkpoint models')
opt = parser.parse_args()

transform = transforms.Compose([
    transforms.ToPILImage()
])

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True

noise_bank = {}
noise_save = None

def train(epoch, perceptual_data_loader):
    global noise_bank
    global noise_save

    model.train()
    loss_print = 0 

    for iteration, batch in enumerate(training_data_loader, 1):
        
        with torch.no_grad():
            features_sum = None
            total_samples = 0
            for num_i, batch_p in enumerate(perceptual_data_loader):
                input_p = batch_p[0]
                input_pc = input_p.cuda()
                Lp, Rp, branch_out_p, features_p = model(input_pc)
                batch_sum = features_p.sum(dim=0) 
                
                if features_sum is None:
                    features_sum = torch.zeros_like(batch_sum)
                
                if features_sum.size() != batch_sum.size():
                    continue
                features_sum += batch_sum
                total_samples += features_p.size(0)

            mean_features = features_sum / total_samples
            mean_features = mean_features.unsqueeze(0)

        if len(noise_bank) != 0:
            noise_id = random.randint(0, len(noise_bank) - 1)
            noise_save = noise_bank[list(noise_bank.keys())[noise_id]]

        input = batch[0]
        input1 = input.cuda()

        name = batch[1]
 
        L1, R1, v1, _ = model(input1)
        N1 = input1 - L1*R1
        
        noise_bank[name] = N1

        if epoch != 1:
            a = torch.ones_like(N1) * 0.5
            mask1 = torch.bernoulli(a)
            mask1 = mask1 * 2 - 1
            mask2 = torch.bernoulli(a)
            mask2 = mask2 * 2 - 1
            n1_b = N1 * mask1
            n2_b = noise_save * mask2
            r = random.random()
            N = r * n1_b + (1 - r) * n2_b

        else:
            a = torch.ones_like(N1) * 0.5
            mask1 = torch.bernoulli(a)
            mask1 = mask1 * 2 - 1
            N = N1 * mask1

        r2 = random.random() - 0.5
        input2 = input1 + N.detach()
        L2, R2, v2, _ = model(input2)
        input3 = (input1 / torch.pow(L1, r2))
        L3, R3, _, _ = model(input3) 

        
        input4 = torch.pow(L1, v1) * R1
        _, _, _, features_L = model(input4) 

        loss1 = rec_loss(input1, L1, R1)
        loss2 = L_loss(input1, L1) 
        loss3 = R_loss(input1, L1, R1)
        loss4 = torch.nn.MSELoss()(R1, R2)
        loss5 = torch.nn.MSELoss()(R1, R3)
        loss6 = kl_loss(features_L, mean_features)

        # overall loss 
        loss = loss1 * 1 + loss2 * 1 + loss3 * 1 + loss4 * 0.75 + loss5 * 0.01 + loss6 * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_G = loss_print + loss.item()

        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_G, optimizer.param_groups[0]['lr']))
            loss_print = 0         

    torch.set_grad_enabled(True)


def checkpoint(epoch):
    model_out_path = opt.weights+"epoch_{}.pth".format(epoch)
    if not os.path.exists(opt.weights):
        os.makedirs(opt.weights)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

print('===> Loading datasets')

test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

perceptual_set = get_training_set(opt.perceptual_test)
perceptual_data_loader = DataLoader(dataset=perceptual_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ')

model= net().cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch, perceptual_data_loader)
    scheduler.step()

    if (epoch) % opt.snapshots == 0:
        checkpoint(epoch)

