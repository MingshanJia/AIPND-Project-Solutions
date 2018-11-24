import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description='Image Predict App')
# set image path 
parser.add_argument('image_path', default='', type=str, help='path of image (default: none)')
# set checkpoint path 
parser.add_argument('checkpoint_path', default='', type=str, help='path of checkpoint')
# set topk
parser.add_argument('--top_k', type=int, default=1, metavar='N', help='topN prediction (default: 1)')
# set showing category 
parser.add_argument('--no_category_names', action='store_true', default=False, help='display category names on/off')
# set json file path  
parser.add_argument('--json_path', default='cat_to_name.json', type=str, help='json file path')
# set gpu usage
parser.add_argument('--no_cuda', action='store_true', default=False, help='gpu on/off (default is on)')

args = parser.parse_args()
# if cuda is available and is allowed
args.cuda = not args.no_cuda and torch.cuda.is_available()

data_dir = 'flowers'
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
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    model.classifier = MyNetwork(checkpoint['input_size'],
                          checkpoint['output_size'],
                          checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

model_loaded = load_checkpoint(args.checkpoint_path)

def process_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img = np.array(img)
    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    
    return img

with open(args.json_path, 'r') as f:
    cat_to_name = json.load(f)

def predict(image_path, model, topk=5):

    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)   # fit to used as input
    if args.cuda:
        model.to('cpu')
        img = img.to('cpu')
    ps = torch.exp(model.forward(img))  
    top_ps, top_labs = ps.topk(topk)
    
    top_ps = top_ps.detach().numpy().tolist()[0]     # make list
    top_labs = top_labs.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[cla] for cla in top_classes]
    
    if args.no_category_names:
        for cla, prob in zip(top_classes, top_ps):
            print("Class No.{} with probability {:.2f}%".format(cla, 100*prob))
    else:
        for cla, prob, name in zip(top_classes, top_ps, top_flowers):
            print("[{}] [No.{}] with probability [{:.2f}%]".format(name, cla, 100*prob))

predict(args.image_path, model_loaded, args.top_k)