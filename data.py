from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image



cam_dict = {"010": "01_0", "050": "05_0", "051": "05_1", "080": "08_0", "081": "08_1", "090": "09_0",
             "110": "11_0", "120": "12_0", "130": "13_0", "140": "14_0", "190": "19_0", "191": "19_1",
             "200": "20_0", "240": "24_0", "041": "04_1"}


class TrainDataset( Dataset):
    def __init__( self , img_list ):
        super(type(self),self).__init__()
        self.img_list = img_list
    def __len__( self ):
        return len(self.img_list)
    def __getitem__( self , idx ):
        #filename processing
        batch = {}
        img_name = self.img_list[idx].split('/')
        fname = img_name[-1]
        img_frontal_name = self.img_list[idx].split('_')
        img_frontal_name[-2] = '051'
        # Normal Illumination
        # img_frontal_name[-1] = '07.png'
        img_frontal_name = '_'.join( img_frontal_name ).split('/')
        img_frontal_name[-3] = '05_1'
        batch['img'] = Image.open( '/'.join( img_name ) )
        batch['img32'] = Image.open( '/'.join( img_name[:-2] + ['32x32' , img_name[-1] ] ) )
        batch['img64'] = Image.open( '/'.join( img_name[:-2] + ['64x64' , img_name[-1] ] ) )
        batch['img_frontal'] = Image.open( '/'.join(img_frontal_name) )
        batch['img32_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['32x32' , img_frontal_name[-1] ] ) )
        batch['img64_frontal'] = Image.open( '/'.join( img_frontal_name[:-2] + ['64x64' , img_frontal_name[-1] ] ) )
        # patch_name_list = ['left_eye','right_eye','nose','mouth']
        # for patch_name in patch_name_list:
        #     batch[patch_name] = Image.open( '/'.join(img_name[:-2] + ['patch' , patch_name , img_name[-1] ]) )
        #     batch[patch_name+'_frontal'] = Image.open( '/'.join(img_frontal_name[:-2] + ['patch' , patch_name , img_frontal_name[-1] ]) )
        totensor = transforms.ToTensor()
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #
        for k in batch:
            batch[k] = normalize(totensor( batch[k] ))
        #     batch[k] = batch[k] *2.0 -1.0
            #
            #if batch[k].max() <= 0.9:
            #    print( "{} {} {}".format( batch[k].max(), self.img_list[idx] , k  ))
            #if batch[k].min() >= -0.9:
            #    print( "{} {} {}".format( batch[k].min() , self.img_list[idx] , k ) )
        label = int( self.img_list[idx].split('/')[-1].split('_')[0] )
        batch['label'] = label
        batch['name'] = fname
        return batch