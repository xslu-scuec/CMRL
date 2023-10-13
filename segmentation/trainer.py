import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim
from dataset.data_loader import data_generator
import numpy as np
import os
from utils import weights_init, dice_loss
from dataset.dataset_lits_val import Val_Dataset
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from utils.sliding_window import SlidingWindow
import torch.nn.functional as F
from tlogger import get_logger
from utils.dice_loss import DC_and_CE_loss
from tqdm import tqdm


def pad_sample(sample, target_shape):
    diff = [max(target_dim - sample_dim, 0) for target_dim, sample_dim in zip(target_shape, sample.shape)]
    padded_sample = np.pad(sample, [(0, d) for d in diff], mode='constant')
    return torch.tensor(padded_sample), diff


def val(model, val_ct, val_mri, device):
    model.eval()
    diceaverage = dice_loss.DiceAverage()
    dices = []
    input_shape = (64, 128, 128)
    with torch.no_grad():
        for idx, sampled_batch in enumerate(val_ct):
            data, target, name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name']
            original_shape = data.shape[2:5]
            padded_image, diff = pad_sample(data.squeeze(0).squeeze(0), input_shape)
            sw = SlidingWindow(padded_image.shape, input_shape, has_batch_dim=True, striding=(32, 64, 64))
            result = torch.zeros((1, 5) + padded_image.shape)
            padded_image = padded_image.unsqueeze(0).unsqueeze(0)
            y_pred_count = torch.zeros_like(result)
            for slicer in sw:
                data_slicer = padded_image[slicer]
                data_slicer = data_slicer.to(device)
                _, output = model(data_slicer, stream_id=1)
                output = F.softmax(output, dim=1)
                result[slicer] += output.to('cpu')
                y_pred_count[slicer] += 1
            result /= y_pred_count
            result_cropped = result[:, :, :original_shape[0], :original_shape[1] - diff[1],
                             :original_shape[2] - diff[2]]
            y_ = torch.argmax(result_cropped, axis=1)
            y_ = np.squeeze(y_.detach().cpu().numpy(), axis=0).astype(np.float32)
            dc = diceaverage.dice(y_, target)
            dices.append(dc)
        dsc_ct = np.mean(dices)

        dices = []
        for idx, sampled_batch in enumerate(val_mri):
            data, target, name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name']
            original_shape = data.shape[2:5]
            padded_image, diff = pad_sample(data.squeeze(0).squeeze(0), input_shape)
            sw = SlidingWindow(padded_image.shape, input_shape, has_batch_dim=True, striding=(32, 64, 64))
            result = torch.zeros((1, 5) + padded_image.shape)
            padded_image = padded_image.unsqueeze(0).unsqueeze(0)
            y_pred_count = torch.zeros_like(result)
            for slicer in sw:
                data_slicer = padded_image[slicer]
                data_slicer = data_slicer.to(device)
                _, output = model(data_slicer, stream_id=0)
                output = F.softmax(output, dim=1)
                result[slicer] += output.to('cpu')
                y_pred_count[slicer] += 1
            result /= y_pred_count
            result_cropped = result[:, :, :original_shape[0], :original_shape[1] - diff[1],
                             :original_shape[2] - diff[2]]
            y_ = torch.argmax(result_cropped, axis=1)
            y_ = np.squeeze(y_.detach().cpu().numpy(), axis=0).astype(np.float32)
            dc = diceaverage.dice(y_, target)
            dices.append(dc)
        dsc_mri = np.mean(dices)
        val_log = OrderedDict({'Val_dice_CT': dsc_ct, 'Val_dice_MRI': dsc_mri})
        return val_log


def train(model, train_dl_ct, train_dl_mri, optimizer_ct, optimizer_mri, loss_func, device):
    model.train()
    losses_ct = []
    losses_mri = []

    data1, target1 = next(train_dl_ct)
    data2, target2 = next(train_dl_mri)

    data1, target1 = torch.from_numpy(data1), torch.from_numpy(target1)
    data2, target2 = torch.from_numpy(data2), torch.from_numpy(target2)

    target1 = target1.long()
    target2 = target2.long()

    data1 = data1.float()
    data2 = data2.float()

    data1, target1 = data1.to(device), target1.to(device)
    data2, target2 = data2.to(device), target2.to(device)

    optimizer_ct.zero_grad()
    _, output_ct = model(data1, 1)
    loss_ct = loss_func(output_ct, target1)
    loss_ct.backward()
    optimizer_ct.step()
    losses_ct.append(loss_ct.item())

    optimizer_mri.zero_grad()
    _, output_mri = model(data2, 0)
    loss_mri = loss_func(output_mri, target2)
    loss_mri.backward()
    optimizer_mri.step()
    losses_mri.append(loss_mri.item())

    return OrderedDict({'loss_ct': np.mean(losses_ct), 'loss_mri': np.mean(losses_mri)})


def trainer(args, model, device, save_path):
    val_mri = Val_Dataset(args.val_MRI)
    val_ct = Val_Dataset(args.val_CT)
    writer = SummaryWriter()
    model.apply(weights_init.init_model)

    optimizer_ct = optim.Adam(model.parameters(), lr=args.lr_ct, eps=1e-05)
    optimizer_mri = optim.Adam(model.parameters(), lr=args.lr_mri, eps=1e-05)

    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    best = [0, 0, 0]
    best_epoch = 0

    if os.path.exists(save_path + 'latest_model.pth'):
        checkpoint = torch.load(save_path + 'latest_model.pth')
        model.load_state_dict(checkpoint['net'])
        optimizer_ct.load_state_dict(checkpoint['optimizer'])
        best_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best = checkpoint['best']
        print("loading checkpoint finished!")
    logger = get_logger(args.logger)
    logger.info("starting train....")
    logger.info(model)
    for epoch in tqdm(range(best_epoch, args.epochs), ncols=70):
        train_dl_mri = data_generator(args.train_MRI, args.batch_size)
        train_dl_ct = data_generator(args.train_CT, args.batch_size)
        for _ in range(7):
            train_log = train(model, train_dl_ct, train_dl_mri, optimizer_ct, optimizer_mri, loss, writer, epoch,
                              device)
            val_log = val(model, val_ct, val_mri, device)
            state = {'net': model.state_dict(), 'optimizer': optimizer_ct.state_dict(), 'epoch': epoch, 'loss': loss,
                     'best': best}
            print('loss_ct:', round(train_log['loss_ct'], 4), 'loss_mri:', round(train_log['loss_mri'], 4))
            if val_log['Val_dice_CT'] + val_log['Val_dice_MRI'] > best[1] + best[2]:
                print('Saving best model')
                torch.save(state, os.path.join(save_path, 'best_model.pth'))
                best[0] = epoch
                best[1] = val_log['Val_dice_CT']
                best[2] = val_log['Val_dice_MRI']
                logger.info('Best dice at Epoch: {} |ct: {} |mri: {}'.format(best[0], best[1], best[2]))
            torch.save(state, os.path.join(save_path, 'latest_model.pth'))
