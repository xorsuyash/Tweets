import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import NeuralNet
from Split import Datasplit

device=torch.device("cuda")
device2=torch.device("cpu")
CUDA_LAUNCH_BLOCKING=1


input_size=768

hidden_zise=100
num_classes=2
num_epochs=1000
batch_size=100
learning_rate=0.001

X=torch.load('embedings2.pt')
y=torch.load(';labels.pt')

for i in range(len(y)):
    if y[i]==4:
        y[i]=1

y=torch.tensor(y,dtype=torch.float32)
X=X.to(device2)




dataset=[]

for i in range(len(X)):
    dataset.append((X[i],y[i]))


split=Datasplit(dataset,shuffle=True)

train_loader,val_loader,test_loader=split.get_split(batch_size=100,num_workers=8)


#model

model=NeuralNet()


criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


#training loop 

for epoch in range(num_epochs):
    model.train()
    train_loss=0.0

    for i,(text,label) in enumerate(train_loader):
        label=label.view(-1,2)



        output=model(text)

        loss=criterion(output,label.view(-1,1))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss+=loss.item()
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)}')


    val_loss=0.0
    model.eval()

    for data,target in val_loader:

        output=model(data)
        loss=criterion(output,target.view(-1,1))
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        val_loss+=loss.item()
    print(f'Epoch {epoch+1} \t\t val_loss: {val_loss / len(val_loader)}')


    




    

        





