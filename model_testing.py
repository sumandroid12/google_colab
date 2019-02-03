import torch
from torch.backends import cudnn
import torch.nn as nn
from light_cnn import LightCNN_29Layers_v2
from data import TrainDataset
import config
from torch.autograd import Variable
from my_utils import *
from utils import *

cudnn.enabled = True
img_list = open('img_list/051_all', 'r').read().split('\n')
img_list.pop()

feature_extract_model = LightCNN_29Layers_v2(num_classes=80013).cuda()
feature_extract_model = torch.nn.DataParallel(feature_extract_model).cuda()

feature_extract_model.module.fc2 = nn.Linear(in_features=256, out_features=360).cuda()
torch.nn.init.kaiming_uniform_(feature_extract_model.module.fc2.weight)

optim_LCNN = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-4, )
resume_model(feature_extract_model,'model_save')
resume_optimizer(optim_LCNN, feature_extract_model,'model_save')
# Train LightCNN on multipie

# input
trainloader = torch.utils.data.DataLoader(TrainDataset(img_list), batch_size=64,
                                          shuffle=True, num_workers=2, pin_memory=True)

last_epoch = 2
cross_entropy = nn.CrossEntropyLoss().cuda()
epoch = 80

for epoch in range(last_epoch + 1, 10):
    top1 = AverageMeter()
    for step, batch in enumerate(trainloader):
        for k in batch:
            if k != 'name':
                batch[k] = Variable(batch[k].cuda(async=True), requires_grad=False)

        feature_extract_model.zero_grad()

        output, _ = feature_extract_model(to_gray(batch['img']))
        loss = cross_entropy(output, batch['label'])

        loss.backward()
        optim_LCNN.step()
        prec1, = accuracy(output.data, batch['label'], (1,))

        top1.update(prec1.item(), batch['label'].size()[0])

        if step % 10 == 0:
            print(top1.avg, step) 
    print('epoch end accuracy: ',top1.avg)
    print(epoch)
    save_model(feature_extract_model, 'multipie_finetune', epoch)
