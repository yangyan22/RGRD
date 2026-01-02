import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
from torch.utils.data import DataLoader
from net.net import net
from data import get_eval_set
from utils import *
from torchvision import transforms
from thop import profile
import time

parser = argparse.ArgumentParser(description='Pytorch RGRD')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--model', default='./weight/RGRD.pth', help='Pretrained base model')
parser.add_argument('--data_test', type=str, default='test-dataset/SICEp2/image/')
parser.add_argument('--output_folder', type=str, default='./results/SICEp2/')
opt = parser.parse_args()

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')
model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

dump_input = torch.ones(1,3,256,256).cuda()
flops, params = profile(model, (dump_input,))
print('flops: ', flops, 'params: ', params)

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()
        print(name)

        with torch.no_grad():
            L, R, v, _ = model(input)
            I = torch.pow(L, v) * R
         
        # os.makedirs(os.path.join(opt.output_folder, 'L'), exist_ok=True)  # save L 
        # os.makedirs(os.path.join(opt.output_folder, 'R'), exist_ok=True)  # save R 
        os.makedirs(os.path.join(opt.output_folder, 'I'), exist_ok=True)
                                
        # L = L.cpu()
        # R = R.cpu()
        I = I.cpu()

        # L_img = transforms.ToPILImage()(L.squeeze(0))
        # R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))

        # L_img.save(opt.output_folder + '/L/' + name[0])
        # R_img.save(opt.output_folder + '/R/' + name[0])
        I_img.save(opt.output_folder + '/I/' + name[0])

    torch.set_grad_enabled(True)

eval()


