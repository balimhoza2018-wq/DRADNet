import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from lib.dradnetModel import Shunted_DRADnet
from Test import test_with_eval

torch.backends.cudnn.enabled = False


def weighted_IoU_Focal_loss(pred, mask_fg):
    """
    Boundary-aware Focal + Weighted IoU Loss (foreground only)
    Uses hardcoded parameters: alpha=0.25, gamma=2.5, kernel_size=31
    """
    alpha = 0.25
    gamma = 2.5
    kernel_size = 31
    
    # ---- Shape safety ----
    if mask_fg.dim() == 3:
        mask_fg = mask_fg.unsqueeze(1)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)

    mask_fg = mask_fg.float()

    # ---- Boundary weight ----
    padding = kernel_size // 2
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask_fg, kernel_size, stride=1, padding=padding) - mask_fg
    )
    weit = weit / (weit.mean(dim=(1,2,3), keepdim=True) + 1e-6)

    # ---- Foreground focal loss ----
    bce_fg = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    pt_fg = torch.exp(-bce_fg)
    focal_fg = alpha * (1 - pt_fg)**gamma * bce_fg
    wfocal_fg = (weit * focal_fg).sum(dim=(1,2,3)) / weit.sum(dim=(1,2,3))

    # ---- Foreground IoU loss ----
    pred_prob_fg = torch.sigmoid(pred)
    inter_fg = (pred_prob_fg * mask_fg * weit).sum(dim=(1,2,3))
    pred_area_fg = (pred_prob_fg * weit).sum(dim=(1,2,3))
    mask_area_fg = (mask_fg * weit).sum(dim=(1,2,3))
    union_fg = pred_area_fg + mask_area_fg - inter_fg
    wiou_fg = 1 - (inter_fg + 1e-6) / (union_fg + 1e-6)

    # ---- Combine foreground losses ----
    return (wfocal_fg + wiou_fg).mean()




def show_tensor(tensor_list,path=None):
    for i, tensor in enumerate(tensor_list):
        if isinstance(tensor,np.ndarray):
            array=tensor
        else:
            array=np.array(tensor.to('cpu'))
        if path is None:
            np.savetxt('tensor_values'+str(i+1)+'.txt', array, fmt='%0.6f')
        else:
            np.savetxt(path, array, fmt='%0.6f')


def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = \
        AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts    = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize),
                                       mode='bilinear', align_corners=True)
                gts    = F.interpolate(gts,    size=(trainsize, trainsize),
                                       mode='bilinear', align_corners=True)
            
            
            # MODEL OUTPUT
            (lateral_map_2_fg, lateral_map_3_fg,
             lateral_map_4_fg, lateral_map_5_fg) = model(images)

            # LOSS CALL 
            loss5 = weighted_IoU_Focal_loss(lateral_map_2_fg, gts)
            loss4 = weighted_IoU_Focal_loss(lateral_map_3_fg, gts)
            loss3 = weighted_IoU_Focal_loss(lateral_map_4_fg, gts)
            loss2 = weighted_IoU_Focal_loss(lateral_map_5_fg, gts)
            loss  = loss2 + loss3 + loss4 + loss5

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:.4f}, '
                  'lateral-4: {:.4f}, lateral-5: {:.4f}]'.format(
                      datetime.now(), epoch, opt.epoch, i, total_step,
                      loss_record2.show(), loss_record3.show(),
                      loss_record4.show(), loss_record5.show()))

    # ---- save checkpoint ----
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if epoch % 10 == 0:
        torch.save(model.state_dict(),
                   save_path + 'DRADNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'DRADNet-%d.pth' % epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='Shunted_DRADnet_res') # TODO: Change the name of the folder to save the model
    parser.add_argument('--model_type', type=str, default='Shunted_DRADnet')
    opt = parser.parse_args()

    # ---- load data ----
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    print("length of train dataset: {}".format(len(train_loader)))
    total_step = len(train_loader)
    print("#"*20, "Start Training", "#"*20)
    #import sys
    #sys.exit(0)
    
    # ---- build model ----
    if opt.model_type == 'Shunted_DRADnet':
        model = Shunted_DRADnet().cuda()
    else:
        raise ValueError('Model Not Found, Shunted_DRADnet should be available')


    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)


    eval_config={
    "datasets": ['CVC-300', 'CVC-ClinicDB'], #ORIGINAL
    #"datasets": ['Kvasir', 'CVC-ClinicDB'],
    "metrics": ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae'],
    "test_size":352,
    }
  
    best_eval_res = [[0, 0, 0], [0, 0, 0]]
    for epoch in tqdm(range(1, opt.epoch), desc="Training Epochs"):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt)
        
        print('Evaluating the model...')
        eval_res = test_with_eval(eval_config, model)
        print('Epoch: %d, Evaluation:\n %s' % (epoch, eval_res))
        
        if eval_res[0][0] + eval_res[1][0] - best_eval_res[0][0] - best_eval_res[1][0] > 0:
        # if eval_res[0][0] - best_eval_res[0][0]  > 0:
            best_eval_res = eval_res
            save_path = 'snapshots/{}/'.format(opt.train_save)
            torch.save(model.state_dict(), save_path + 'best.pth')
            print('[Saving Snapshot:]', save_path + 'best.pth in epoch %d' % (epoch + 1))
        
        

            

