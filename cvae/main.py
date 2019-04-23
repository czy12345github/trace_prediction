from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


parser = argparse.ArgumentParser(description='CVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_set = torch.Tensor(np.loadtxt('trainset'))
test_set = torch.Tensor(np.loadtxt('testset'))
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)

num_of_samples = 200

num_of_z = 60
num_of_cond = 77 * 4
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(385+num_of_cond, 200)
        self.fc21 = nn.Linear(200, num_of_z)
        self.fc22 = nn.Linear(200, num_of_z)
        self.fc3 = nn.Linear(num_of_z+num_of_cond,200)
        self.fc4 = nn.Linear(200, 385)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, zc):
        h3 = self.relu(self.fc3(zc))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        cond = x[:,385:]
        zc = torch.cat([z, cond], 1)
        return self.decode(zc), mu, logvar


model = CVAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        inputs = torch.cat([data, data[:,:num_of_cond]], 1)
        if args.cuda:
            data = data.cuda()
            inputs = inputs.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        data = Variable(data)
        inputs = torch.cat([data, data[:,:num_of_cond]], 1)
        if args.cuda:
            data = data.cuda()
            inputs = inputs.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def write_tensor(txt, sample):
    rows, cols = sample.size()
    for i in range(rows):
        line = []
        for j in range(cols):
            line.append(str(int(sample[i,j])))
        line = ' '.join(line) + '\n'
        txt.write(line)

def test_prediction(epoch):
    txt = open('test_epoch'+str(epoch), 'w')
    for i in range(test_set.size()[0]):
        cond = test_set[i,:num_of_cond].view(-1,num_of_cond)
        x = cond
        for j in range(num_of_samples-1):
            cond = torch.cat([cond,x], 0)
        sample = Variable(torch.randn(num_of_samples, num_of_z))
        cond = Variable(cond)
        sample = torch.cat([sample, cond], 1)
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).view(-1,77)
        sample = sample.data.max(1,keepdim=True)[1]
        sample = sample.view(-1,5) + 1
        write_tensor(txt, sample)
    txt.close()

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch > 48:
        test_prediction(epoch)
