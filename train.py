import torch
from torchvision import transforms
import config
from data import TrainDataset
from network import Discriminator, Generator
from log import TensorBoardX
from utils import *
from my_utils import *
from light_cnn import LightCNN_29Layers_v2
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torchsummary import summary
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

if __name__ == "__main__":
    cudnn.enabled = True
    img_list = open(config.train['img_list'], 'r').read().split('\n')
    img_list.pop()

    # input
    trainloader = torch.utils.data.DataLoader(TrainDataset(img_list), batch_size=config.train['batch_size'],
                                              shuffle=True, num_workers=2, pin_memory=True)
    G = Generator(zdim=config.G['zdim'], use_batchnorm=config.G['use_batchnorm'],
                  use_residual_block=config.G['use_residual_block'],
                  num_classes=config.G['num_classes']).cuda()
    D = Discriminator(use_batchnorm=True).cuda()
    feature_extract_model = LightCNN_29Layers_v2(num_classes=80013).cuda()
    feature_extract_model = torch.nn.DataParallel(feature_extract_model).cuda()

    feature_extract_model.module.fc2 = torch.nn.Linear(in_features=256, out_features=360).cuda()
    torch.nn.init.kaiming_uniform_(feature_extract_model.module.fc2.weight)

    resume_model(feature_extract_model, 'multipie_finetune_model')
    feature_extract_model.eval()
    set_requires_grad(feature_extract_model, False)

    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=1e-5)
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=1e-4)
    last_epoch = -1

    tb = TensorBoardX(config_filename_list=["config.py"], sub_dir='./adversarial_wgan')

    l1_loss = torch.nn.L1Loss().cuda()
    mse = torch.nn.MSELoss().cuda()
    cross_entropy = torch.nn.CrossEntropyLoss().cuda()
    cosine_loss = torch.nn.CosineEmbeddingLoss().cuda()
    BCE_loss = torch.nn.BCELoss().cuda()

    real_label = 0.9
    fake_label = 0

    set_requires_grad(D, False)
    feature_extract_model.eval()
    label = torch.full((config.train['batch_size'], 1, 2, 2), fake_label).cuda()

    for epoch in range(last_epoch + 1, config.train['num_epochs']):
        for step, batch in enumerate(trainloader):
            if batch['img'].shape[0] != 5:
                continue
            for k in batch:
                if k != 'name':
                    batch[k] = Variable(batch[k].cuda(async=True), requires_grad=False)

            # Train Discriminator
            # if step % 5 == 0:
            set_requires_grad(D, True)
            D.zero_grad()
            label.fill_(fake_label)
            # Train Discriminator with fake Data
            z = torch.randn((len(batch['img']), config.G['zdim'])).cuda()
            z = (z * 2) - 1
            img128_fake, img64_fake, img32_fake = G(batch['img'], batch['img64'], batch['img32'], z)
            D_out_fake = D(img128_fake.detach())
            D_err_fake = BCE_loss(D_out_fake, label)
            D_err_fake.backward()

            # Train Discriminator with real Data
            label.fill_(real_label)
            D_out_real = D(batch['img_frontal'])
            D_err_real = BCE_loss(D_out_real, label)
            D_err_real.backward()

            L_D_adv = D_err_real.item() + D_err_fake.item()

            optimizer_D.step()
            set_requires_grad(D, False)

            # Train Generator
            G.zero_grad()
            label.fill_(real_label)

            _, real_features = feature_extract_model(to_gray(batch['img_frontal']))
            _, fake_features = feature_extract_model(to_gray(img128_fake))

            output = D(img128_fake)
            # print(output.shape,label.unsqueeze(1).shape)
            G_err_adv = BCE_loss(output, label)
            tb.add_scalar('loss_G_ADV', G_err_adv, len(trainloader) * epoch + step, 'train')
            one = torch.ones(5, 1).cuda()
            G_feature_loss = cosine_loss(real_features, fake_features, one)
            tb.add_scalar('loss_G_Feature', G_feature_loss, len(trainloader) * epoch + step, 'train')

            L_128 = l1_loss(img128_fake, batch['img_frontal'])
            L_64 = l1_loss(img64_fake, batch['img64_frontal'])
            L_32 = l1_loss(img32_fake, batch['img32_frontal'])
            L_rec = config.loss['weight_128'] * L_128 + config.loss['weight_64'] * L_64 + config.loss[
                'weight_32'] * L_32
            tb.add_scalar('loss_G_reconstruction', L_rec, len(trainloader) * epoch + step, 'train')

            # Symmetry Loss
            inv_idx128 = torch.arange(img128_fake.size()[3] - 1, -1, -1).long().cuda()
            img128_fake_flip = img128_fake.index_select(3, Variable(inv_idx128))
            img128_fake_flip.detach_()
            inv_idx64 = torch.arange(img64_fake.size()[3] - 1, -1, -1).long()   .cuda()
            img64_fake_flip = img64_fake.index_select(3, Variable(inv_idx64))
            img64_fake_flip.detach_()
            inv_idx32 = torch.arange(img32_fake.size()[3] - 1, -1, -1).long().cuda()
            img32_fake_flip = img32_fake.index_select(3, Variable(inv_idx32))
            img32_fake_flip.detach_()
            symmetry_128_loss = l1_loss(img128_fake, img128_fake_flip)
            symmetry_64_loss = l1_loss(img64_fake, img64_fake_flip)
            symmetry_32_loss = l1_loss(img32_fake, img32_fake_flip)
            symmetry_loss = config.loss['weight_128'] * symmetry_128_loss + config.loss[
                'weight_64'] * symmetry_64_loss + config.loss['weight_32'] * symmetry_32_loss
            tb.add_scalar('loss_G_symmetry', symmetry_loss, len(trainloader) * epoch + step, 'train')


            G_err_total = G_err_adv * config.loss['weight_adv_G'] + L_rec * 5 + symmetry_loss * config.loss[
                'weight_symmetry'] + config.loss['weight_identity_preserving'] * G_feature_loss
            G_err_total.backward()
            optimizer_G.step()

            tb.add_scalar('D_err_real', D_err_real, step, 'train')
            tb.add_scalar('D_err_real', D_err_real, step, 'train')

            if step % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (config.train['num_epochs'], epoch, step, len(trainloader),
                         L_D_adv, G_err_adv.item(), D_out_real.mean().item(), D_out_fake.mean().item(),
                         output.mean().item()))

            if step % 100 == 0:
                tb.add_image_grid('Predict', 4, img128_fake.data.float() / 2.0 + 0.5, step, 'train')
                tb.add_image_grid('input', 4, batch['img'].data.float() / 2.0 + 0.5, step, 'train')

