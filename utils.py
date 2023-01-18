from math import floor, ceil
from unet import UNET

import torchvision.transforms as transforms
import torch.nn as nn

import torch


# Load a pre-trained model
def load_model(filepath):
    model = UNET(3, 2).cpu()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    return model


# Generates the segmentation mask of an image
def create_segmentation_mask(model, image, tensor_transform, DEVICE, threshold):
    with torch.no_grad():
        logits = tensor_transform(model(image.to(DEVICE)))
        feature_map = nn.Softmax(dim=1)(logits)
        mask = feature_map.detach().cpu().squeeze(0).numpy()[1, :, :]
    return mask > threshold


# Transform an image into a square image of size 'size x size'.
# Returns the transformed image and a sequence of transform operation
# to reverse to the original image size
def transform_image(image, size):
    tr = transforms.ToTensor()
    image_tensor = tr(image)

    # Removing alpha value, if any;
    image_tensor = image_tensor[:3, :, :]

    original_size = image_tensor.shape

    if image_tensor.shape[1] == image_tensor.shape[2]:
        # Images are square, only resizing
        tr = transforms.Resize(size=(size, size))
        untr = transforms.Resize(size=(original_size[1], original_size[2]))
        return tr(image_tensor).unsqueeze(dim=0), untr

    # Images are not square, first resizing with a max
    # length of size then adding padding
    tr = transforms.Resize(size=size-1, max_size=size)
    image_tensor = tr(image_tensor)

    if image_tensor.shape[1] > image_tensor.shape[2]:
        padding = (image_tensor.shape[1] - image_tensor.shape[2]) / 2
        tr = transforms.Pad((ceil(padding), 0, floor(padding), 0))
        untr = transforms.Compose([
            transforms.Resize((original_size[1], original_size[1])),
            transforms.CenterCrop((original_size[1], original_size[2]))
        ])
    else:
        padding = (image_tensor.shape[2] - image_tensor.shape[1]) / 2
        tr = transforms.Pad((0, ceil(padding), 0, floor(padding)))
        untr = transforms.Compose([
            transforms.Resize((original_size[2], original_size[2])),
            transforms.CenterCrop((original_size[1], original_size[2]))
        ])
    return tr(image_tensor).unsqueeze(dim=0), untr
