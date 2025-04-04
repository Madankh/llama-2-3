import  torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

_ = torch.manual_seed(0)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# load the MNIST dataset
minst_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(minst_train, batch_size=10, shuffle=True)

# Load the MNIST test set
minst_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
test_loader = torch.utils.data.DataLoader(minst_test, batch_size=10, shuffle=True)

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoraNet(nn.Module):
    def __init__(self, hidden_size = 1000, hidden_size_2=2000):
        super(LoraNet, self).__init__()
        self.linear1 = nn.linear(28*28, hidden_size)
        self.linear2 = nn.linear(hidden_size, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
    def forward(self, img):
        x = img.view(-1, 28*28);
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
net = LoraNet().to(device)

def train(train_loader, net , epochs=5, total_iteration_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    total_iter = 0
    for epoch in range(epochs):
        loss_sum = 0
        num_iterations = 0
        data_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iteration_limit is not None:
            data_iter.total = total_iteration_limit
        
        for data in data_iter:
            num_iterations += 1
            total_iter += 1
            x, y = data
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = net.train(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            arvg_loss = loss_sum/num_iterations
            # data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iteration_limit is not None and total_iter >= total_iteration_limit:
                return
train(train_loader, net, epochs=5)

original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()


            