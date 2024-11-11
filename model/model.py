import torch
from torch import nn, optim
from utils import utils
from tqdm import tqdm
from utils import params


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False, attention=False):
        super().__init__()
        self.outermost = outermost
        self.attention = attention
        if input_c is None:
            input_c = nf

        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.2)]
            model = down + [submodule] + up

        if attention:
            model += [CBAM(nf)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
      if self.outermost:
          return self.model(x)
      else:
          return torch.cat([x, self.model(x)], 1)
      
class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64, attention=False):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True, attention=attention)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True, attention=attention)
        out_filters = num_filters * 8
        for _ in range(2):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block, attention=attention)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True, attention=attention)
    
    def forward(self, x):
        return self.model(x)
    
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3): #changed filters from 64 to 128
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,  # changed stride from 2
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        # if act: layers += [nn.LeakyReLU(0.2, True)]
        if act: layers += [nn.GELU()]  # changed from LeakyRelu
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction = reduction
        outChannels = max(channels // reduction, 1)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, outChannels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(outChannels, channels, kernel_size=1, stride=1, padding=0)

        # Spatial Attention
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)

        # Spatial Attention
        #spatial_attention = self.sigmoid(self.conv1(torch.max(x, dim=1, keepdim=True)[0]))
        spatial_attention = self.sigmoid(self.conv1(x))

        # Apply attention
        x = x * channel_attention * spatial_attention
        

        return x   
    

class MainModel(nn.Module):
    def __init__(self, net_G=None, lambda_L1=100., attention=False):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = utils.init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64, attention=attention), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = utils.init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = utils.GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    # def optimize(self, lr_G=2e-5, lr_D=2e-7, beta1=0.5, beta2=0.999):
    #     self.forward()
    #     self.net_D.train()
    #     self.set_requires_grad(self.net_D, True)
    #     self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))  # Move optimizer initialization here
    #     self.opt_D.zero_grad()
    #     self.backward_D()
    #     self.opt_D.step()
        
    #     self.net_G.train()
    #     self.set_requires_grad(self.net_D, False)
    #     self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))  # Move optimizer initialization here
    #     self.opt_G.zero_grad()
    #     self.backward_G()
    #     self.opt_G.step()
    
    def optimize_discriminator(self, lr_G=2e-5, lr_D=2e-7, beta1=0.5, beta2=0.999):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))  # Move optimizer initialization here
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
    
    def optimize_generator(self, lr_G=2e-5, lr_D=2e-7, beta1=0.5, beta2=0.999):
        self.forward()    
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))  # Move optimizer initialization here
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
    

def train(model, train_dl, val_dl, epochs, generator_steps=1, discriminator_steps=1, display_every=2000): # changed display_every from 200
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = utils.create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        # log_results(loss_meter_dict)
        for data in tqdm(train_dl):
            model.setup_input(data)
            # model.optimize()  # Update both generator and discriminator in one call
            # Update Generator
            for _ in range(generator_steps):
                model.optimize_generator()
            # Update discriminator
            for _ in range(discriminator_steps):
                model.optimize_discriminator()

            utils.update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            # log_results(loss_meter_dict)
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                utils.log_results(loss_meter_dict) # function to print out the losses
                utils.visualize(model, data, save=False) # function displaying the model's outputs
        utils.log_results(loss_meter_dict)
        print("==> epochs: ",e+1,"/",epochs)
