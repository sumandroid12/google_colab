import torch
from torch.backends import cudnn
from log import TensorBoardX
import torch.nn as nn
from light_cnn import LightCNN_29Layers_v2
from data_ex_feat import TrainDataset
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

resume_model(feature_extract_model, 'multipie_finetune_model')
feature_extract_model.eval()
# Train LightCNN on multipie

# input
trainloader = torch.utils.data.DataLoader(TrainDataset(img_list), batch_size=8,
                                          shuffle=True, num_workers=2, pin_memory=True)

last_epoch = 1
cross_entropy = nn.CrossEntropyLoss().cuda()
epoch = 1

tb = TensorBoardX(sub_dir='multipie_finetune')
embeddings = torch.zeros(200,8,256)
# labels = torch.zeros(28,8)
with torch.no_grad():
    labels = []
    for step, batch in enumerate(trainloader):
        if step == 200:
            break
        for k in batch:
            if k != 'name':
                batch[k] = Variable(batch[k].cuda(async=True), requires_grad=False)

        output, features = feature_extract_model(to_gray(batch['img']))
        embeddings[step] = features.detach()
        # labels[step] = batch['label'].detach()
        labels += batch['name']

    embeddings = embeddings.reshape(-1, 256,)
    # labels = labels.reshape(-1)
    print(embeddings.shape, len(labels))
    tb.writer['test'].add_embedding(embeddings, labels,)
