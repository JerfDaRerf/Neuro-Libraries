"""
Implements a style transfer in PyTorch.
"""

import torch
import torch.nn as nn

# Compute the scalar content loss for style transfer.
def content_loss(content_weight, content_current, content_original):

    squared_error = (content_current - content_original) ** 2
    
    # Sum the squared error over all elements
    loss = content_weight * torch.sum(squared_error)
    
    return loss
    
# Compute the Gram matrix from features.
def gram_matrix(features, normalize=True):
    gram = None

    # Get dimensions of features
    N = features.size(dim=0)
    C = features.size(dim=1)
    H = features.size(dim=2)
    W = features.size(dim=3)

    # Gram = F*F^T
    F = torch.reshape(features, (N, C, H * W))
    F_t = torch.permute(F, (0, 2, 1))
    gram = torch.matmul(F[:], F_t[:])

    # Normalize if necessary - if True, divide the Gram matrix by the number of neurons (H * W * C)
    if(normalize):
        gram = gram / (C * H * W)

    return gram

# Computes the style loss at a set of layers. 
# Returns a PyTorch Tensor holding a scalar giving the style loss
def style_loss(feats, style_layers, style_targets, style_weights):
    
    style_loss = 0
    i = 0
    for layer in style_layers:
      style_loss += style_weights[i] * torch.sum(torch.pow(gram_matrix(feats[layer]) - style_targets[i], 2))
      i += 1

    return style_loss

# Compute total variation loss.
def tv_loss(img, tv_weight):
    
    # Horizontal variation for all channels
    horizontal_var = (img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2
    # Vertical variation for all channels
    vertical_var = (img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2
    
    # Sum over all differences and all channels, then weight by tv_weight
    loss = tv_weight * (torch.sum(horizontal_var) + torch.sum(vertical_var))
    
    return loss

# Compute the guided Gram matrix from features by applying the 
# regional guidance mask to its corresponding feature.

def guided_gram_matrix(features, masks, normalize=True):

  # Get dimensions of features
  N = features.size(dim=0)
  R = features.size(dim=1)
  C = features.size(dim=2)
  H = features.size(dim=3)
  W = features.size(dim=4)

  # Rearrange features to be in form: (C, N, R, H, W)
  featuresC = torch.permute(features, (2, 0, 1, 3, 4))

  # Apply masks across each channel
  featuresMasked = torch.permute(featuresC[:] * masks, (1, 2, 0, 3, 4))

  # Prep for gram = F*F^T
  F = torch.reshape(featuresMasked, (N, R, C, H * W))
  F_t = torch.permute(F, (0, 1, 3, 2))
  gram = torch.matmul(F[:], F_t[:])

  # Normalize if necessary
  if(normalize):
    gram = gram / (C * H * W)

  return gram

# Computes the style loss at a set of layers.
def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):

    style_loss = 0.0
    i = 0
    for layer in style_layers:
        toAdd = style_weights[i] * torch.sum(torch.pow(guided_gram_matrix(feats[layer], content_masks[layer]) - style_targets[i], 2))
        style_loss += toAdd
        i += 1

    return style_loss

