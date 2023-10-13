from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class SlidingWindow(object):
    """SlidingWindow

    Sliding window iterator which produces slice objects to slice in a sliding window. This is useful for inference.

    """
    def __init__(self, img_shape, window_shape, has_batch_dim=True, striding=None):
        """Constructs a sliding window iterator

        Parameters
        ----------
        img_shape : tuple or list
            shape of the image to slide over
        window_shape : tuple or list
            shape of the window to extract
        striding : tuple or list, optional
            amount to move the window between each position
        """
        self.img_shape = img_shape
        self.window_shape = window_shape
        self.rank = len(img_shape)
        self.curr_pos = [0] * self.rank
        self.end_pos = [0] * self.rank
        self.done = False
        self.striding = window_shape
        self.has_batch_dim = has_batch_dim
        if striding:
            self.striding = striding

    def __iter__(self):
        return self
    
    # py 2.* compatability hack
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.done:
            raise StopIteration()
        slicer = [slice(None)] * (self.rank + 2) if self.has_batch_dim else [slice(None)] * self.rank
        move_dim = True
        for dim, pos in enumerate(self.curr_pos):
            low = pos
            high = pos + self.window_shape[dim]
            if move_dim:
                if high >= self.img_shape[dim]:
                    self.curr_pos[dim] = 0
                    move_dim = True
                else:
                    self.curr_pos[dim] += self.striding[dim]
                    move_dim = False
            if high >= self.img_shape[dim]:
                low = self.img_shape[dim] - self.window_shape[dim]
                high = self.img_shape[dim]

            if self.has_batch_dim:
                slicer[dim + 2] = slice(low, high)
            else:
                slicer[dim] = slice(low, high)

        if (np.array(self.curr_pos) == self.end_pos).all():
            self.done = True
        return slicer
