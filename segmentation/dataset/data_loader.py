from typing import Tuple

import numpy as np
import pandas as pd
from segmentation.dataset.transforms import RandomCrop
from segmentation.dataset.augmentation import gaussian_noise, gaussian_offset


def load_volfile(data_file: str):
    assert data_file.endswith(('.npz')), f"unknown data file: {data_file}"
    data = np.load(data_file)
    image = data['data'][0]
    label = data['data'][1]
    return image, label


def data_generator(vol_pairs, batch_size, is_train=True) -> Tuple:
    filename = pd.read_csv(vol_pairs, dtype=str, header=None).values.tolist()
    np.random.shuffle(filename)
    num_iter = 0
    num_max = len(filename)
    while num_iter < num_max:
        batch_images, batch_labels = [], []
        for _ in range(batch_size):
            image, label = load_volfile(filename[num_iter][0])
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            image, label = RandomCrop(64)(image, label)
            if is_train:
                image = gaussian_noise(image, sigma=0.1)
                image = gaussian_offset(image, sigma=0.1)
            batch_images.append(image)
            batch_labels.append(label)
            num_iter += 1
            if num_iter >= num_max:
                break
        batch_images = np.stack(batch_images, axis=0)
        batch_labels = np.stack(batch_labels, axis=0)
        yield (batch_images, batch_labels)
