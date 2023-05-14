import torch 
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):

        super(NeuralNet,self).__init__()

        #self.l1=nn.Linear(input_size,768)
        self.dropout=torch.nn.Dropout(0.3)
        self.l1=nn.Linear(768,1)
        #self.l3=nn.Linear(500,100)

        #self.relu=nn.ReLU()

        #self.l4=nn.Linear(100,1)


    def forward(self,x):

        out=self.dropout(x)
        out=self.l1(out)
       # out=self.relu(out)
       # out=self.l2(out)
       # out=self.relu(out)
        #out=self.l3(out)
       # out=self.relu(out)
        #out=self.l4(out)

        return out 
    





















