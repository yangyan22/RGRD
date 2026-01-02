import torch
import torch.nn as nn


class L_net(nn.Module):
    def __init__(self, num=32):
        super(L_net, self).__init__()

        self.main_part1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
        )

        self.main_part2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),             
            nn.Linear(num, 1),        
            nn.Sigmoid()             
        )

    def forward(self, input):
        features = self.main_part1(input)      
        main_out = self.main_part2(features)   
        branch_out = self.branch(features)     
        return torch.sigmoid(main_out), branch_out, features


class R_net(nn.Module):
    def __init__(self, num=32):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),         
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),          
            nn.ReLU(),  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),          
            nn.ReLU(),                  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),          
            nn.ReLU(),  
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),          
            nn.ReLU(),             
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input))


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.L_net = L_net(num=32)
        self.R_net = R_net(num=32)

    def forward(self, input):
        L, branch_out, features = self.L_net(input)
        R = self.R_net(input)
        return L, R, branch_out, features
