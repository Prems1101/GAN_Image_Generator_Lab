import torch
import torch.nn as nn

device = "cpu"

# ===============================
# Vanilla GAN
# ===============================

class VanillaGenerator(nn.Module):
    def __init__(self, z_dim=100, img_dim=3*64*64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(True),

            nn.Linear(256,512),
            nn.ReLU(True),

            nn.Linear(512,1024),
            nn.ReLU(True),

            nn.Linear(1024,img_dim),
            nn.Tanh()
        )

    def forward(self,z):
        img=self.model(z)
        img=img.view(z.size(0),3,64,64)
        return img


class VanillaDiscriminator(nn.Module):
    def __init__(self,img_dim=3*64*64):
        super().__init__()

        self.model=nn.Sequential(
            nn.Linear(img_dim,1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),

            nn.Linear(512,256),
            nn.LeakyReLU(0.2),

            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,img):
        img=img.view(img.size(0),-1)
        return self.model(img)


# ===============================
# DCGAN
# ===============================

class DCGenerator(nn.Module):

    def __init__(self,nz=100,ngf=32,nc=3):
        super().__init__()

        self.main=nn.Sequential(

            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self,x):
        return self.main(x)


class DCDiscriminator(nn.Module):

    def __init__(self,ndf=32,nc=3):
        super().__init__()

        self.main=nn.Sequential(

            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x).view(-1)

class CGenerator(nn.Module):

    def __init__(self,nz=100,ngf=32,nc=3,num_classes=3):

        super().__init__()

        self.label_emb=nn.Embedding(num_classes,num_classes)

        self.main=nn.Sequential(

            nn.ConvTranspose2d(nz+num_classes,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self,noise,labels):

        label_input=self.label_emb(labels).unsqueeze(2).unsqueeze(3)

        x=torch.cat([noise,label_input],1)

        return self.main(x)

class CDiscriminator(nn.Module):

    def __init__(self, ndf=32, nc=3, num_classes=3):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(

            nn.Conv2d(nc + num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):

        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_input = label_input.repeat(1, 1, img.size(2), img.size(3))

        x = torch.cat([img, label_input], 1)

        return self.main(x).view(-1)