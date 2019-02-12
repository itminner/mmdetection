"""Convert other dataset to custom dataset format.
    Its all images should locate in one single folder, and the Annotation file format is:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
"""

from __future__ import division
import argparse
import os


class DatasetConverter(object):
    def __init__(self, _dataset_path, _output_path):
        self.dataset_path = _dataset_path
        self.output_path = _output_path
        self.total_img_num = 0

    def convert(self):
        pass

    def __call__(self):
        self.convert()
        print('Convert done. total image num: ', self.total_img_num)
