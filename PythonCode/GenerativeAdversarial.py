from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96

# Generate a PyTorch Tensor of random noise from Gaussian distribution.
def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    
    noise = None
    noise = torch.randn((batch_size, noise_dim), dtype=dtype, device=device) 
  
    return noise

# Builds a PyTorch model implementing a discriminator architecture.
def discriminator():

    model = None
    model = nn.Sequential(nn.Linear(in_features=784, out_features=400),
                          nn.LeakyReLU(negative_slope = 0.05),
                          nn.Linear(in_features=400, out_features=200),
                          nn.LeakyReLU(negative_slope = 0.05),
                          nn.Linear(in_features=200, out_features=100),
                          nn.LeakyReLU(negative_slope = 0.05),
                          nn.Linear(in_features=100, out_features=1))

    return model

# Builds a PyTorch model implementing a generator architecture
def generator(noise_dim=NOISE_DIM):
    model = None
    model = nn.Sequential(nn.Linear(in_features=noise_dim, out_features=128),
                          nn.ReLU(),
                          nn.Linear(in_features=128, out_features=256),
                          nn.ReLU(),
                          nn.Linear(in_features=256, out_features=512),
                          nn.ReLU(),
                          nn.Linear(in_features=512, out_features=784),
                          nn.Tanh())

    return model

# Computes discriminator loss using cross entropy function.
def discriminator_loss(logits_real, logits_fake):
    loss = None
   
    # Loss for the real and fake data
    loss = (nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones(logits_real.size()[0], dtype=torch.float, device = logits_real.device)))\
     + (nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros(logits_real.size()[0], dtype=torch.float, device=logits_fake.device)))  

    return loss

# Computes the generator loss
def generator_loss(logits_fake):
    loss = None
    loss = (nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones(logits_fake.size()[0], dtype = torch.float, device = logits_fake.device)))
    
    return loss

# Construct and return an Adam optimizer for the model with learning rate 1e-3, beta1=0.5, and beta2=0.999.
def get_optimizer(model):

    optimizer = None   
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, betas = (0.5, 0.999))

    return optimizer

#Trains a GAN. 
def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250, 
  
  iter_count = 0
  for epoch in range(num_epochs):
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      # Iteration of training the discriminator.
      
      d_total_error = None
      D_solver.zero_grad()
      x = x.to(device)
      x_normalized = torch.flatten(x, start_dim= 1)
      x_normalized -= torch.min(x_normalized)
      x_normalized /= torch.max(x_normalized)

      logits_real = D(x_normalized * 2 - 1)
      d_noise = sample_noise(batch_size, noise_size, device = device)
      fake_images = G(d_noise).detach()
      logits_fake = D(fake_images)
      
      logits_real = torch.flatten(logits_real, start_dim = 0)
      logits_fake = torch.flatten(logits_fake, start_dim = 0)
      d_total_error = discriminator_loss(logits_real, logits_fake)
      d_total_error.backward()
      D_solver.step()
      # Training of the generator now
      
      g_error = None
      fake_images = None
      G_solver.zero_grad()
      g_noise= sample_noise(batch_size, noise_size, device = device)
      fake_images = G(g_noise)

      gen_logits_fake = D(fake_images)
      gen_logits_fake = torch.flatten(gen_logits_fake, start_dim = 0)
      g_error = generator_loss(gen_logits_fake)
      g_error.backward()
      G_solver.step()
      
      #Prints iteration count 
      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      return imgs_numpy    



# Uses PyTorch to build a DCGAN discriminator. 
def build_dc_classifier():

    return nn.Sequential(nn.Unflatten(dim = 1, unflattened_size=(1, 28, 28)),
                         nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
                         nn.LeakyReLU(negative_slope=0.01),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
                         nn.LeakyReLU(negative_slope=0.01),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Flatten(),
                         nn.Linear(in_features=4*4*64, out_features=4*4*64),
                         nn.LeakyReLU(negative_slope=0.01),
                         nn.Linear(in_features=4*4*64, out_features=1),)
   

# Uses PyTorch to build a DCGAN generator.
def build_dc_generator(noise_dim=NOISE_DIM):

    return nn.Sequential(nn.Linear(in_features=noise_dim, out_features=1024),
                        nn.ReLU(),
                        nn.BatchNorm1d(num_features=1024),
                        nn.Linear(in_features=1024, out_features=7*7*128),
                        nn.ReLU(),
                        nn.BatchNorm1d(num_features=7*7*128),
                        nn.Unflatten(dim = 1, unflattened_size=(128, 7, 7)),
                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=64),
                        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
                        nn.Tanh(),
                        nn.Flatten())
