import os.path as osp
import os
import torch.optim as optim
import random
import numpy as np
import torch as tc
import gc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import random
import cv2
import torch.nn.functional as F
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim import  AdamW
import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from tap import Tap
import gc
from PIL import Image
import numpy as np
import scipy.stats as st
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class InferenceArgumentParser(Tap):
    segment_id: str ='20230925002745'
    segment_path:str='./eval_scrolls'
    model_path:str= 'outputs/vesuvius/pretraining_all/vesuvius-models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
    out_path:str='./'
    stride: int = 2
    start_idx:int=15
    workers: int = 25
    batch_size: int = 512
    size:int=64
    reverse:int=0
args = InferenceArgumentParser().parse_args()
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'

    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = args.size
    tile_size = args.size
    stride = args.stride

    train_batch_size = args.batch_size # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 50 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 2

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 5

    print_freq = 50
    num_workers = args.workers

    seed = 42

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=90,shift_limit=0.1,scale_limit=0.1,p=0.5),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=2, distort_limit=0.3, p=0.4),
        A.CoarseDropout(max_holes=5, max_width=int(size * 0.05), max_height=int(size * 0.05), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy import ndimage
from scipy.ndimage import zoom
def read_image_mask(fragment_id,start_idx=18,end_idx=38,rotation=0):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)
    # idxs = range(0, 65)
    dataset_path=args.segment_path
    for i in idxs:
        
        image = cv2.imread(f"{dataset_path}/{fragment_id}/layers/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image=np.clip(image,0,200)

        images.append(image)
    images = np.stack(images, axis=2)
    if fragment_id in ['20230925002745','20230926164853'] or args.reverse==1:
        images=images[:,:,::-1]
    fragment_mask=None
    if os.path.exists(f'{dataset_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask=cv2.imread(CFG.comp_dataset_path + f"{dataset_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        kernel = np.ones((16,16),np.uint8)
        fragment_mask = cv2.erode(fragment_mask,kernel,iterations = 1)
    return images,fragment_mask
def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    image,fragment_mask = read_image_mask(fragment_id,s,e,rotation)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask
def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):

        image, mask,fragment_mask = read_image_mask(fragment_id)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))
        
                if fragment_id == CFG.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                        train_images.append(image[y1:y2, x1:x2])
                        train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.kernel=gkern(64,2)
        self.kernel/=self.kernel.max()
        self.kernel=torch.FloatTensor(self.kernel)
    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)

            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # offset=random.choice([0,1,2,3,4])
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)
            
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")
        # for l in self.convs:
        #     for m in l._modules:
        #         init_weights(m)
    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

 

from collections import OrderedDict
def normalization(x):
    """input.shape=(batch,f1,f2,...)"""
    #[batch,f1,f2]->dim[1,2]
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    return (x-mean)/(std+1e-9)


from i3dall import InceptionI3d
class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=224,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        # self.backbone=SegModel(model_depth=50)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        # self.loss_func2= smp.losses.FocalLoss(mode='binary',gamma=2)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15)
        # self.loss_func=nn.HuberLoss(delta=5.0)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        # self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=700)
        if self.hparams.enc=='resnet34':
            self.backbone = generate_model(model_depth=34, n_input_channels=1,forward_features=True,n_classes=700)
            state_dict=torch.load('./r3d34_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='resnest101':
            self.backbone = generate_model(model_depth=101, n_input_channels=1,forward_features=True,n_classes=1039)
            state_dict=torch.load('./r3d101_KM_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='2p1d':
            self.backbone = generate_2p1d(model_depth=34, n_input_channels=1,n_classes=700)
            state_dict=torch.load('./r2p1d34_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1_s.weight']
            state_dict['conv1_s.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='wide50':
            self.backbone = generate_wide(model_depth=50, n_input_channels=1,n_classes=700,forward_features=True,k=2)
        elif self.hparams.enc=='i3d':
            self.backbone=InceptionI3d(in_channels=1,num_classes=512)
        elif self.hparams.enc=='resnext101':
            self.backbone=resnext101(sample_size=112,
                                  sample_duration=16,
                                  shortcut_type='B',
                                  cardinality=32,
                                  num_classes=600)
            state_dict = torch.load('./kinetics_resnext_101_RGB_16_best.pth')['state_dict']
            checkpoint_custom = OrderedDict()
            for key_model, key_checkpoint in zip(self.backbone.state_dict().keys(), state_dict.keys()):
                checkpoint_custom.update({f'{key_model}': state_dict[f'{key_checkpoint}']})

            self.backbone.load_state_dict(checkpoint_custom, strict=True)
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        else:
            self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=700)
            state_dict=torch.load('./r3d50_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # outputs=torch.clip(outputs,min=-1,max=1)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        # print(loss1)
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/Accuracy_macro", acc_macro,on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   
def TTA(x:tc.Tensor,model:nn.Module):
    #x.shape=(batch,c,h,w)
    shape=x.shape
    x=[x,*[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)],]
    x=tc.cat(x,dim=0)
    x=model(x)
    # x=torch.sigmoid(x)
    # print(x.shape)
    x=x.reshape(4,shape[0],CFG.size//4,CFG.size//4)
    
    x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x=tc.stack(x,dim=0)
    return x.mean(0)
def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    model.eval()
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()
    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            y_preds = model(images)
            # y_preds =TTA(images,model)
        # y_preds = y_preds.to('cpu').numpy()

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    # mask_pred/=mask_pred.max()
    return mask_pred

fragments=os.listdir('./eval_scrolls')
fragment_id=args.segment_id

test_loader,test_xyxz,test_shape,fragment_mask=get_img_splits(fragment_id,args.start_idx,args.start_idx+30,0)
model=RegressionPLModel.load_from_checkpoint(args.model_path,strict=False)
model.cuda()
model.eval()
mask_pred= predict_fn(test_loader, model, device, test_xyxz,test_shape)
mask_pred=np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
mask_pred/=mask_pred.max()
mask_pred=(mask_pred*255).astype(np.uint8)
mask_pred=Image.fromarray(mask_pred)
mask_pred.save(f'{args.out_path}/{fragment_id}_{args.stride}_{args.start_idx}.png')