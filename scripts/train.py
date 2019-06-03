import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import PIL.Image
from torchvision import transforms, models


from model import BasementModel

def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasementModel()#kara

    model.to(device).eval()

    print("Done loading model.")
    return model


def train(model, target, content_features, style_features, style_grams, epochs, optimizer ):
    
    show_every = int(args.epochs / 10)
    
    #training
    for ii in range(1, epochs+1):
    
        # get the features from your target image
        target_features = get_features(target, model)
    
        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = make1dim_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
        
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
    
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())

def load_image(img_path, max_size=400, shape=None):    
    image = PIL.Image.open(img_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def make1dim_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    make1dim = torch.mm(tensor, tensor.t())    
    return make1dim


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--input_image_name', type=str, default='input_image.jpg', help='input image name (default: input_image.jpg)')
    parser.add_argument('--reference_image_name', type=str, default='reference_image.jpg', help='random seed (default: reference_image.jpg)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
        
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
    #read vgg model
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    
    # load in content and style image
    content = load_image(os.path.join(args.data_dir, args.input_image_name)).to(device)
    # Resize style to match content, makes code easier
    style = load_image(os.path.join(args.data_dir, args.reference_image_name), shape=content.shape[-2:]).to(device)

    #read content
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: make1dim_matrix(style_features[layer]) for layer in style_features}
    target = content.clone().requires_grad_(True).to(device)
    
    #set weights for 
    style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta
    
    optimizer = torch.optim.Adam([target], lr=0.003)
    
    #training
    train(vgg, target, content_features, style_features, style_grams, args.epochs, optimizer )
    
    #save the result = target
    pilImg = PIL.Image.fromarray(np.uint8(im_convert(target)*256))
    pilImg.save(os.path.join(args.model_dir,'result.jpg'))
