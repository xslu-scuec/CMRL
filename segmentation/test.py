import argparse
import os
import SimpleITK as sitk
import numpy as np
import torch
from dataset.dataset_lits_val import Val_Dataset
import torch.nn.functional as F
from utils.sliding_window import SlidingWindow
from models.CMRL_model_3D import CMRL_model


def pad_sample(sample, target_shape):
    diff = [max(target_dim - sample_dim, 0) for target_dim, sample_dim in zip(target_shape, sample.shape)]
    padded_sample = np.pad(sample, [(0, d) for d in diff], mode='constant')
    return torch.tensor(padded_sample), diff


def prediction_on_modal(args, model, dataloader, stream_id, device):
    model.eval()
    input_shape = (64, 128, 128)
    if stream_id == 1:
        modal = "Synapse"
    else:
        modal = "CHAOS"
    with torch.no_grad():
        for idx, sampled_batch in enumerate(dataloader):
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
                features, output = model(data_slicer, stream_id=stream_id)
                output = F.softmax(output, dim=1)
                result[slicer] += output.to('cpu')
                y_pred_count[slicer] += 1
            result /= y_pred_count
            result_cropped = result[:, :, :original_shape[0], :original_shape[1] - diff[1],
                             :original_shape[2] - diff[2]]
            y_ = torch.argmax(result_cropped, axis=1)
            y_ = np.squeeze(y_.detach().cpu().numpy(), axis=0).astype(np.float32)
            save_path = os.path.join(args.save_path, 'fold_' + args.fold, args.checkpoint, modal)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img = sitk.GetImageFromArray(y_)
            sitk.WriteImage(img, os.path.join(save_path, name + '.nii.gz'))


def inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CMRL_model(in_channels=1, num_classes=5, base_c=32).to(device)
    model.to(device)
    checkpoint_path = os.path.join(r"/media/lab429/data1/CYCY/MultiModal/Data/trained_models/fold_" + args.fold,
                                   args.checkpoint + "_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        best_epoch1 = checkpoint['epoch']
        print('best_epoch:', best_epoch1)
    test_ct = Val_Dataset(args.test_ct)
    test_mri = Val_Dataset(args.test_mri)
    prediction_on_modal(args, model, test_ct, 1, device)
    prediction_on_modal(args, model, test_mri, 0, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dual-stream encode-decoder network for multi-model learning")
    parser.add_argument("--model_dir", type=str, default='checkpoint', help="directory of saving model")
    parser.add_argument("--model_checkpoint", type=int, default=2000, help="load model checkpoint")
    parser.add_argument("--logger", type=str, default='./test.log', help="path for train logger")
    parser.add_argument("--checkpoint", type=str, default='best', help="path for train logger")
    parser.add_argument('--test_ct',
                        default=r"/media/lab429/data1/CYCY/MultiModal/Data/Synapse/fold_4/test_ct.csv")
    parser.add_argument('--test_mri',
                        default=r'/media/lab429/data1/CYCY/MultiModal/Data/CHAOS/fold_4/test_mri.csv')
    parser.add_argument('--save_path', default=r'/media/lab429/data1/CYCY/MultiModal/output')
    parser.add_argument('--fold', type=str, default='4')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    inference(args)
