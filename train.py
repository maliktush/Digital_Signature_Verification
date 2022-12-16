import os
from data import get_data_loader
from PIL import ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from metrics import accuracy
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#
torch.manual_seed(2000)
torch.cuda.manual_seed_all(2000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def step(loss,optimizer):
    loss.backward()
    optimizer.step()

def train(model, optimizer, criterion, dataloader):
    model.train()
    t_loss = 0
    n = 0

    for batch, (datax1, datax2, datay) in enumerate(dataloader):
        datax1 = datax1.to(device)
        datax2 = datax2.to(device)
        datay = datay.to(device)
        optimizer.zero_grad()
        datax1, datax2 = model(datax1, datax2)
        loss = criterion(datax1, datax2, datay)
        step(loss,optimizer)

        n += len(datax1)
        t_loss += loss.item() * len(datax1)
        if (batch + 1) % 100 == 0 or batch == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch + 1, len(dataloader), t_loss / n))
            t_loss = 0
            n = 0


@torch.no_grad()
def get_loss_and_accuracy(model, criterion, dataloader):
    model.eval()
    t_loss = 0
    n = 0

    distances = []

    for batch, (datax1, datax2, datay) in enumerate(dataloader):
        datax1 = datax1.to(device)
        datax2 = datax2.to(device)
        datay = datay.to(device)

        outputx1, outputx2 = model(datax1, datax2)
        loss = criterion(outputx1, outputx2, datay)
        distances.extend(zip(torch.pairwise_distance(outputx1, outputx2, 2).cpu().tolist(), datay.cpu().tolist()))

        n += len(outputx1)
        t_loss += loss.item() * len(outputx1)

        if (batch + 1) % 100 == 0 or batch == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch + 1, len(dataloader), t_loss / n))

    distances, datay = zip(*distances)
    distances, datay = torch.tensor(distances), torch.tensor(datay)
    max_accuracy = accuracy(distances, datay)
    print(f'Accuracy: {max_accuracy}')
    return t_loss / n, max_accuracy


if __name__ == "__main__":

    batch_size = 64
    lr = 1e-5
    num_epochs = 15

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, choices=['cedar', 'bengali', 'hindi'], default='cedar')
    parser.add_argument('--model', type=str, choices=['model1', 'model2', 'model3'], default='model3')
    args = parser.parse_args()

    if args.model == 'model1':
      from model1 import SigNet, ContrastiveLoss
    elif args.model == 'model2':
      from model2 import SigNet, ContrastiveLoss
    else :
      from model3 import SigNet, ContrastiveLoss

    model = SigNet().to(device)

    criterion = ContrastiveLoss(1, 3, 1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, eps=1e-8,weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    image_transform = transforms.Compose([
        transforms.Resize((155, 220)),
        ImageOps.invert,
        transforms.ToTensor(),
    ])

    trainloader = get_data_loader(True, args.batch_size, image_transform,args.dataset)
    testloader = get_data_loader(False,args.batch_size,image_transform,args.dataset)

    os.makedirs('checkpoints',exist_ok=True)

    model.train()
    print(model)
    results=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Training Phase')
        train(model, optimizer, criterion, trainloader)

        print('Evaluating Phase')
        loss, acc = get_loss_and_accuracy(model, criterion, testloader)
        scheduler.step()
        results.append(acc*100)

        c = {
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optim': optimizer.state_dict(),
        }
        if epoch%5 == 0 or epoch==num_epochs-1:
            print('Checkpoint')
            torch.save(c, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))
    
    x=[i for i in range(1,num_epochs+1)]
    # plotting
    plt.title("Epochs vs Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(top=105)
    plt.ylim(bottom=0) 
    plt.plot(x, results, color="green")
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.savefig('epochs.png')
