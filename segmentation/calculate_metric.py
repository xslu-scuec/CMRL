import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def precision(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        precision = binary.precision(pred, gt)
        return precision
    else:
        return 0


def recall(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        recall = binary.recall(pred, gt)
        return recall
    else:
        return 0


def assd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        assd = binary.assd(pred, gt)
        return assd
    else:
        return 0


def process_label(label):
    liver = label == 1
    spleen = label == 2
    right_kidney = label == 3
    left_kidney = label == 4

    return liver, spleen, right_kidney, left_kidney


def inference():
    label_list = sorted(glob.glob(os.path.join(
        r"/media/lab429/data1/CYCY/MultiModal/Data/CHAOS/Spacing_8/fold_3/test_nii",
        '*nii.gz')))
    dest_file = r"/media/lab429/data1/CYCY/Result/Fourth/MultiModal/spacing8/fold_3_spac8_UNet1_Res/CHAOS"
    infer_list = sorted(
        glob.glob(os.path.join(dest_file, '*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)

    Dice_liver = []
    Dice_spleen = []
    Dice_right_kidney = []
    Dice_left_kidney = []

    recall_liver = []
    recall_spleen = []
    recall_right_kidney = []
    recall_left_kidney = []

    precision_liver = []
    precision_spleen = []
    precision_right_kidney = []
    precision_left_kidney = []

    assd_liver = []
    assd_spleen = []
    assd_right_kidney = []
    assd_left_kidney = []

    file = dest_file
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + '/result_metrics.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_liver, label_spleen, label_right_kidney, label_left_kidney = process_label(label)
        infer_liver, infer_spleen, infer_right_kidney, infer_left_kidney = process_label(infer)

        Dice_liver.append(dice(infer_liver, label_liver))
        Dice_spleen.append(dice(infer_spleen, label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney, label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney, label_left_kidney))

        recall_liver.append(recall(infer_liver, label_liver))
        recall_spleen.append(recall(infer_spleen, label_spleen))
        recall_right_kidney.append(recall(infer_right_kidney, label_right_kidney))
        recall_left_kidney.append(recall(infer_left_kidney, label_left_kidney))

        precision_liver.append(precision(infer_liver, label_liver))
        precision_spleen.append(precision(infer_spleen, label_spleen))
        precision_right_kidney.append(precision(infer_right_kidney, label_right_kidney))
        precision_left_kidney.append(precision(infer_left_kidney, label_left_kidney))

        assd_liver.append(assd(infer_liver, label_liver))
        assd_spleen.append(assd(infer_spleen, label_spleen))
        assd_right_kidney.append(assd(infer_right_kidney, label_right_kidney))
        assd_left_kidney.append(assd(infer_left_kidney, label_left_kidney))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))

        fw.write('recall_liver: {:.4f}\n'.format(recall_liver[-1]))
        fw.write('recall_spleen: {:.4f}\n'.format(recall_spleen[-1]))
        fw.write('recall_right_kidney: {:.4f}\n'.format(recall_right_kidney[-1]))
        fw.write('recall_left_kidney: {:.4f}\n'.format(recall_left_kidney[-1]))

        fw.write('precision_liver: {:.4f}\n'.format(precision_liver[-1]))
        fw.write('precision_spleen: {:.4f}\n'.format(precision_spleen[-1]))
        fw.write('precision_right_kidney: {:.4f}\n'.format(precision_right_kidney[-1]))
        fw.write('precision_left_kidney: {:.4f}\n'.format(precision_left_kidney[-1]))

        fw.write('assd_liver: {:.4f}\n'.format(assd_liver[-1]))
        fw.write('assd_spleen: {:.4f}\n'.format(assd_spleen[-1]))
        fw.write('assd_right_kidney: {:.4f}\n'.format(assd_right_kidney[-1]))
        fw.write('assd_left_kidney: {:.4f}\n'.format(assd_left_kidney[-1]))

        dsc = []
        recalls = []
        precisions = []
        assds = []

        dsc.append(Dice_liver[-1])
        dsc.append(Dice_spleen[-1])
        dsc.append(Dice_right_kidney[-1])
        dsc.append(Dice_left_kidney[-1])
        fw.write('DSC:' + str(np.mean(dsc)) + '\n')

        recalls.append(recall_liver[-1])
        recalls.append(recall_spleen[-1])
        recalls.append(recall_right_kidney[-1])
        recalls.append(recall_left_kidney[-1])
        fw.write('recall:' + str(np.mean(recalls)) + '\n')

        precisions.append(precision_liver[-1])
        precisions.append(precision_spleen[-1])
        precisions.append(precision_right_kidney[-1])
        precisions.append(precision_left_kidney[-1])
        fw.write('precision:' + str(np.mean(precisions)) + '\n')

        assds.append(assd_liver[-1])
        assds.append(assd_spleen[-1])
        assds.append(assd_right_kidney[-1])
        assds.append(assd_left_kidney[-1])
        fw.write('assd:' + str(np.mean(assds)) + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_liver' + str(np.mean(Dice_liver)) + '\n')
    fw.write('Dice_spleen' + str(np.mean(Dice_spleen)) + '\n')
    fw.write('Dice_right_kidney' + str(np.mean(Dice_right_kidney)) + '\n')
    fw.write('Dice_left_kidney' + str(np.mean(Dice_left_kidney)) + '\n')

    fw.write('Mean_recall\n')
    fw.write('recall_liver' + str(np.mean(recall_liver)) + '\n')
    fw.write('recall_spleen' + str(np.mean(recall_spleen)) + '\n')
    fw.write('recall_right_kidney' + str(np.mean(recall_right_kidney)) + '\n')
    fw.write('recall_left_kidney' + str(np.mean(recall_left_kidney)) + '\n')

    fw.write('Mean_precision\n')
    fw.write('precision_liver' + str(np.mean(precision_liver)) + '\n')
    fw.write('precision_spleen' + str(np.mean(precision_spleen)) + '\n')
    fw.write('precision_right_kidney' + str(np.mean(precision_right_kidney)) + '\n')
    fw.write('precision_left_kidney' + str(np.mean(precision_left_kidney)) + '\n')

    fw.write('Mean_assd\n')
    fw.write('assd_liver' + str(np.mean(assd_liver)) + '\n')
    fw.write('assd_spleen' + str(np.mean(assd_spleen)) + '\n')
    fw.write('assd_right_kidney' + str(np.mean(assd_right_kidney)) + '\n')
    fw.write('assd_left_kidney' + str(np.mean(assd_left_kidney)) + '\n')

    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    fw.write('dsc:' + str(np.mean(dsc)) + '\n')

    recalls = []
    recalls.append(np.mean(recall_liver))
    recalls.append(np.mean(recall_spleen))
    recalls.append(np.mean(recall_right_kidney))
    recalls.append(np.mean(recall_left_kidney))
    fw.write('recall:' + str(np.mean(recalls)) + '\n')

    precisions = []
    precisions.append(np.mean(precision_liver))
    precisions.append(np.mean(precision_spleen))
    precisions.append(np.mean(precision_right_kidney))
    precisions.append(np.mean(precision_left_kidney))
    fw.write('precision:' + str(np.mean(precisions)) + '\n')

    assds = []
    assds.append(np.mean(assd_liver))
    assds.append(np.mean(assd_spleen))
    assds.append(np.mean(assd_right_kidney))
    assds.append(np.mean(assd_left_kidney))
    fw.write('assd:' + str(np.mean(assds)) + '\n')

    print('done')


if __name__ == '__main__':
    inference()
