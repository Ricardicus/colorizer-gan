import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, channels_in, channels_out, downsampling=False, upsampling=False, lastLayer=False):
        super().__init__()

        downconv = nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = nn.BatchNorm2d(channels_out)

        uprelu = nn.ReLU()
        upnorm = nn.BatchNorm2d(channels_out)
        upconv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1, bias=False)

        if downsampling:
            model = [downconv, downrelu, downnorm]
            self.model = nn.Sequential(*model)
        elif upsampling:
            model = [uprelu, upconv, upnorm, nn.Dropout(0.5)]
            self.model = nn.Sequential(*model)
        elif lastLayer:
            model = [uprelu, upconv, nn.Tanh()]
            self.model = nn.Sequential(*model)
        else:
            raise ValueError("The UNet block must either be downsampling or upsamplig, or be the last layer")

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, complexity=30):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.additional_complexity = complexity

        self.inc = nn.Sequential(*[
            nn.Conv2d(n_channels, self.additional_complexity, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ])

        self.down1 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down2 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down3 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down4 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down5 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)

        self.up1 = UNetBlock(self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up2 = UNetBlock(2*self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up3 = UNetBlock(2*self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up4 = UNetBlock(2*self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up5 = UNetBlock(2*self.additional_complexity, self.additional_complexity, lastLayer = True)

        self.outc = nn.Sequential(*[
            nn.Conv2d(self.additional_complexity, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        ])

        #self.apply(init_weights)

        self.debug = False#True

    def forward(self, x):

        x = self.inc(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        u1 = self.up1(x5)
        u2 = self.up2(torch.cat([u1, x4], dim=1))
        u3 = self.up3(torch.cat([u2, x3], dim=1))
        u4 = self.up4(torch.cat([u3, x2], dim=1))
        u5 = self.up5(torch.cat([u4, x1], dim=1))

        x = self.outc(u5)

        return x

class Discriminator(nn.Module):
    def __init__(self, n_channels=3, complexity=30, dimension=128):
        super(Discriminator, self).__init__()

        self.n_channels = n_channels

        self.additional_complexity = complexity

        self.inc = nn.Sequential(*[
            nn.Conv2d(n_channels, self.additional_complexity, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ])

        self.down1 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down2 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)
        self.down3 = UNetBlock(self.additional_complexity, self.additional_complexity, downsampling = True)

        self.up1 = UNetBlock(self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up2 = UNetBlock(2*self.additional_complexity, self.additional_complexity, upsampling = True)
        self.up3 = UNetBlock(2*self.additional_complexity, self.additional_complexity, lastLayer = True)

        # Final fully connected layer to get a single output
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.additional_complexity * dimension * dimension, 1),
            #nn.BatchNorm1d(1),
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        #print(x[0][0])

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        u1 = self.up1(x3)
        u2 = self.up2(torch.cat([u1, x2], dim=1))
        u3 = self.up3(torch.cat([u2, x1], dim=1))

        # Flatten and pass through the fully connected layer
        x = self.fc(u3)
        x = x / self.additional_complexity

        return self.activation(x)

if __name__ == "__main__":
    # Instantiate the model
    model = Autoencoder()
    print(model)
