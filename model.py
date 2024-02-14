import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    pass
    #if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #    nn.init.xavier_uniform_(m.weight)
        #if m.bias is not None:
        #    nn.init.constant_(m.bias, 0)
    #elif isinstance(m, nn.BatchNorm2d):
    #    nn.init.constant_(m.weight, 1)
    #    nn.init.constant_(m.bias, 0)

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



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 256, kernel_size=3, stride=1, padding=1
        )  # Output: [batch_size, 64, 32, 32]
        self.conv2 = nn.Conv2d(
            256, 128, kernel_size=3, stride=2, padding=1
        )  # Output: [batch_size, 32, 16, 16]
        self.conv3 = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1
        )  # Output: [batch_size, 16, 16, 16]
        self.conv4 = nn.Conv2d(
            64, 3, kernel_size=3, stride=1, padding=1
        )  # Output: [batch_size, 3, 16, 16]

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(
            3, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Output: [batch_size, 16, 32, 32]
        self.deconv2 = nn.ConvTranspose2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # Output: [batch_size, 32, 32, 32]
        self.deconv3 = nn.ConvTranspose2d(
            128, 256, kernel_size=3, stride=1, padding=1
        )  # Output: [batch_size, 64, 32, 32]
        self.deconv4 = nn.ConvTranspose2d(
            256, 3, kernel_size=3, stride=1, padding=1
        )  # this brings it back to [batch_size, 3, 32, 32]

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(
            self.deconv4(x)
        )  # Use sigmoid to ensure output is between 0 and 1, like an image
        return x


if __name__ == "__main__":
    # Instantiate the model
    model = Autoencoder()
    print(model)
