import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from trainer import trainer
import torch
from segmentation.models.CMRL_model_3D import CMRL_model
import os
import argparse

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--num_classes', default=5)
    parser.add_argument('--resume', default=True)
    parser.add_argument('--verbose', default=False)
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--train_MRI',
                        default=r"/media/lab429/data1/CYCY/MultiModal/Data/CHAOS/fold_0/train_mri.csv")
    parser.add_argument('--train_CT',
                        default=r"/media/lab429/data1/CYCY/MultiModal/Data/Synapse/fold_0/train_ct.csv")
    parser.add_argument('--val_MRI',
                        default=r"/media/lab429/data1/CYCY/MultiModal/Data/CHAOS/fold_0/val_mri.csv")
    parser.add_argument('--val_CT',
                        default=r"/media/lab429/data1/CYCY/MultiModal/Data/Synapse/fold_0/val_ct.csv")
    parser.add_argument('--save_path', '-p', default=r'/media/lab429/data1/CYCY/MultiModal/Data/trained_models')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--gpu', default='0', help='use cpu only')
    parser.add_argument("--logger", type=str, default='./log/train.log', help="path for train logger")  # 日志记录地址
    parser.add_argument('--lr_ct', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_mri', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.0001)')
    args = parser.parse_args()
    save_path = os.path.join(args.save_path, 'fold_' + args.fold)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.logger = os.path.join(args.save_path, 'fold_' + args.fold, 'train.log')
    model = CMRL_model(in_channels=1, num_classes=args.num_classes, base_c=32).to(device)
    trainer(args, model, device, save_path)
