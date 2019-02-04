import  torch.nn as nn
import torch.functional as F
import torch
import torch.optim as optim
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.imageChannels=1 #todo find this optimum value
        self.layer1=nn.Conv2d( self.imageChannels,128,[10,10])
        self.layer2=nn.Conv2d(128,64,[9,9])
        self.layer3=nn.Conv2d(64,32,[8,8])
        self.layer4 = nn.Conv2d(32, 16, [7,7])
        self.layer5 = nn.Conv2d(16, 4, [6,6])
        self.layer6 = nn.Conv2d(4, 16, [6, 6])
        self.layer7 = nn.Conv2d( 16, 32, [7,7])
        self.layer8 = nn.Conv2d(32, 64, [8,8])
        self.layer9 = nn.Conv2d(64, 128, [9,9])
        self.layer10 = nn.Conv2d(128, self.imageChannels, [10,10])
        return

    def forward(self, x):
        x=self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x=self.layer10(self.layer9(self.layer8(self.layer7(self.layer6(x)))))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.imageChannels = 1  # todo find this optimum value
        self.layer1 = nn.Conv2d(self.imageChannels, 128, [10, 10])
        self.layer2 = nn.Conv2d(128, 64, [9, 9])
        self.layer3=nn.Conv2d(64,32,[8,8])
        self.layer4 = nn.Conv2d(32, 16, [7,7])
        self.layer5 = nn.Conv2d(16, 4, [6,6])
        self.layer6 = nn.Conv2d(4,1,[2,2])
        self.layer7=nn.Linear(1234,1)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x=self.layer6(x)
        x=x.view(-1,1)
        return self.layer7(x)


G= Generator()
D= Discriminator()
g_optimizer = optim.Adam(G.parameters(),lr=0.01)
d_optimizer = optim.Adam(D.parameters(),lr=0.01)

g_loss=0 # todo find the log likelihood function in pytorch
d_loss = 0

def train_generator(G,D,epoch=3):
    for i in range(epoch):
        g_optimizer.zero_grad()
        g_input_data= get_g_input()
        g_output=G(g_input_data) # or XAB
        d_output=D(g_output)
        error_g= g_loss() #log ( 1 - d_output)
        error_g.backward()
        g_optimizer.step()


def train_discriminator(G,D,epoch=3):
    for i in range(epoch):
        g_optimizer.zero_grad()
        real_data=get_real_data()
        d_output_real=D(real_data)

        g_input_data= get_g_input()
        g_output_fake=G(g_input_data) # or XAB
        d_output_fake=D(g_output_fake)


        error_d= g_loss() # - log ( 1 - d_output_fake ) - log( d_output_real)
        error_d.backward()
        d_optimizer.step()




