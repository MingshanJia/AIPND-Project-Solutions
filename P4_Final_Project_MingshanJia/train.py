import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Image Classifier Train App')
# set train data directory
parser.add_argument('data_dir', default='', type=str, help='path to data directory (default: none)')
# set directory to save checkpoints
parser.add_argument('--save_dir', default='save', type=str, help='set directory to save checkpoints (default: save)')
# set epochs
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
# set gpu usage
parser.add_argument('--no_cuda', action='store_true', default=False, help='gpu on/off (default is on)')
# set learning rate
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
# set neural network architecture
parser.add_argument('--arch', default='vgg16', type=str, help='chose architecture vgg16 or vgg13')
# set hidden layers for classifier
parser.add_argument('--hidden_units', default=[4096, 1000, 400], dest='hidden_layers', nargs='+', type=int, help='list of hidden layers')

args = parser.parse_args()
# if cuda is available and is allowed
args.cuda = not args.no_cuda and torch.cuda.is_available()

data_dir = args.data_dir          #set dir from args
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms)

trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
testloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)

#set arch from args
if args.arch == 'vgg16':     
    model = models.vgg16(pretrained=True) 
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True) 
else:
    print("currently you can only choose between vgg13 and vgg16")
         

class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()      
        self.hidden_layers=nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])          
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)  
        self.dropout = nn.Dropout(p=drop_p)
      
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)   
        return F.log_softmax(x, dim=1)

for param in model.parameters():
    param.requires_grad = False
    
my_classifer = MyNetwork(25088, 102, args.hidden_layers, drop_p=0.5)      # set hidden network to do
model.classifier = my_classifer

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)     # set learning rate from args

def validation(model, validloaders, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloaders:
        if args.cuda:
            model.to('cuda')
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item()    
        equality = (labels.data == outputs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def train_model(model, trainloaders, epochs, print_every, criterion, optimizer, device = 'cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloaders):
            steps += 1
            if args.cuda:
                model.to('cuda')
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss, accuracy = validation(model, validloaders, criterion)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.3f}".format(running_loss/print_every),
                     "Valid Loss: {:.3f}".format(test_loss/len(validloaders)),
                     "Valid Accuracy :{:.3f}%".format(100*accuracy/len(validloaders)))

                running_loss = 0
                model.train()
                           
train_model(model, trainloaders, args.epochs, 40, criterion, optimizer, 'gpu')  

def test_model(model, testloaders):
    model.eval()
    accuracy = 0  
    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloaders):
            if args.cuda:
                model.to('cuda')
                images, labels = images.to('cuda'), labels.to('cuda')
            
            outputs = model.forward(images)
            _, predicted = outputs.max(dim=1)
            equality = labels.data == predicted
            
            if ii == 0:
                print(predicted)      # idx of predicted class
                print(torch.exp(_))   # probability of prediction
                print(equality)       
            
            accuracy += equality.type(torch.FloatTensor).mean()
            #print(accuracy)
    print("Prediction accuracy in test set is {:.3f}%".format(100*accuracy/len(testloaders)))
    
test_model(model, testloaders)

# save trained model
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'features': model.features,
              'input_size': 25088, 
              'output_size': 102,
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              }

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(checkpoint, args.save_dir + '/checkpoint.pth')      # set dir to save checkpoint