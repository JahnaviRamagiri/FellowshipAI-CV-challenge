import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Train:
    def __init__(self, model, device, train_loader, optimizer, l1):
        
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.l1 = l1

        self.train_loss = []

        self.train_acc = []
        self.train_endacc = []
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)


            self.optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.criterion(y_pred, target)

            l1_regularization = torch.tensor(0., requires_grad=True)
            for param in self.model.parameters():
                l1_regularization = l1_regularization + torch.norm(param, p=1)

            loss = loss + self.l1 * l1_regularization

            self.train_loss.append(loss)

            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100 * correct/processed)
        self.train_endacc.append(self.train_acc[-1])