import torch
from torch import nn
from torchvision.models import vgg16
import pyro
from pyro import distributions as dist

class Encoder(nn.Module):
    def __init__(self, z_dim, image_channels):
        super(Encoder, self).__init__()
        self.vgg16 = vgg16()
        self.vgg16.features[0] = nn.Conv2d(image_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16.classifier.add_module('7',nn.ReLU(inplace=True))
        self.linear_loc = nn.Linear(in_features=1000, out_features=z_dim, bias=True)
        self.linear_scale = nn.Linear(in_features=1000, out_features=z_dim, bias=True)
        self.softplus = nn.Softplus()
    def forward(self, x):
        output =  self.vgg16(x)
        loc, scale = self.linear_loc(output), self.softplus(self.linear_scale(output))
        return loc, scale
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim, image_channels):
        super(Decoder, self).__init__()
        self.anti_classifier = nn.Sequential(
                                nn.Linear(in_features=z_dim, out_features=1000),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=1000, out_features=4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(in_features=4096, out_features=512*7*7),
                                nn.ReLU(inplace=True)
                               )
#reshape from [N, 25088] to [N, 512, 7, 7]
        self.anti_features = nn.Sequential(
                              nn.ConvTranspose2d(512, 512, 3, 2),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(512,512, 3, 1, 1),
                              nn.ReLU(inplace=True),
                              #nn.Conv2d(512, 512, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              #nn.Conv2d(512, 512, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(512, 512, 3, 2),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(512,256, 3, 1, 1),
                              nn.ReLU(inplace=True),
                              #nn.Conv2d(512, 512, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              #nn.Conv2d(512, 256, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(256,256, 3, 2),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(256, 128, 3, 1, 1),
                              nn.ReLU(inplace=True),
                              #nn.Conv2d(256, 256, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              #nn.Conv2d(256, 128, 3, 1, 1),
                              #nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(128,128, 3, 2),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(128,64, 3, 2),
                              nn.ReLU(inplace=True),
                              nn.ConvTranspose2d(64, image_channels, 3, 2,output_padding=1)
                             )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.anti_classifier(z)
        out = out.view((-1, 512, 7, 7))
        out = self.anti_features(out)
        out = self.sigmoid(out)
        return out
    

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, image_channels=1, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, image_channels)
        self.decoder = Decoder(z_dim, image_channels)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            #print(loc_img.shape)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(3), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
