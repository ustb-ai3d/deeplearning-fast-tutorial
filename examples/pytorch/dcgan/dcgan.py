from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import Generator, Discriminator

num_epochs = 20
batch_size = 64
image_size = 64     # cifar-10 image size
nz = 100            # noise size
nc = 3              # image channels
ngf = 64            # generator feature map size
ndf = 64            # discriminator feature map size
lr = 2e-4           # learning rate
beta1 = 0.5         # Adam hyperparam
device = torch.device('cuda:0')
dataset = dset.CIFAR10(
    root='/tmp/datasets/',
    download=True,
    transform=transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0)

netG = Generator(nc, ngf, nz).to(device)
netD = Discriminator(nc, ndf).to(device)


criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(1, num_epochs + 1):
    for i, (images, _) in enumerate(loader):
        #### fDx ####
        netD.zero_grad()
        # train with real data
        real = images.to(device)
        # last batch may have different size
        b_size = real.size(0)
        
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake data
        label.fill_(fake_label)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        # detach gradients here so that gredients of G won't be updated
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_fake + errD_real
        optimizerD.step()

        #### fGx ####
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# save models
torch.save(netG.state_dict(), '/tmp/netG.pth')
torch.save(netD.state_dict(), '/tmp/netD.pth')

# show figure
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()