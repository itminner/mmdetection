"""Convert supervisely annotation tool dataset to custom dataset format.
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
import warnings
import argparse
import shutil
import os
import json

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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


class SuperviselyDatasetConverter(DatasetConverter):
    IMAGE_FORMATS = ('bmp', 'jpg', 'bmp', 'png')
    LABEL_DICT = {'person': 0, 'head': 1, 'tabel_card': 2, 'useless': 3}
    total_img_num = 0

    def get_image_name_by_ann_file_path(self, ann_file_name, img_dir_path):
        for img_format in SuperviselyDatasetConverter.IMAGE_FORMATS:
            img_name = ann_file_name.replace('.json', '.' + img_format)
            if os.path.exists(os.path.join(img_dir_path, img_name)):
                return img_name
        warnings.warn("unsupport image format !!!  " + ann_file_name, Warning)
        return None

    def convert_one_ann_json_format(self, ann_file_path, img_filename, ann_items):
        ann_item = {}

        f = open(ann_file_path, encoding='utf-8')
        origin_json = json.load(f)
        ann_item['filename'] = img_filename

        # image size
        img_size = origin_json['size']
        ann_item['width'] = img_size['height']
        ann_item['height'] = img_size['width']

        # objects
        bboxes = []
        bboxes_ignore = []
        labels = []
        ann = {}
        ann['bboxes'] = bboxes
        ann['labels'] = labels
        ann['bboxes_ignore'] = bboxes_ignore

        objects = origin_json['objects']
        for ann_obj in objects:
            points = ann_obj['points']
            lt_x, lt_y = points['exterior'][0]
            lt_x = int(lt_x)
            lt_y = int(lt_y)
            rb_x, rb_y = points['exterior'][1]
            rb_x = int(rb_x)
            rb_y = int(rb_y)
            label_txt = ann_obj['classTitle']

            if label_txt != 'useless':
                bboxes.extend((lt_x, lt_y, rb_x - lt_x, rb_y - lt_y))
                if label_txt not in SuperviselyDatasetConverter.LABEL_DICT.keys():
                    print("unrecognize class title: ", label_txt)
                labels.append(SuperviselyDatasetConverter.LABEL_DICT[label_txt])
            else:
                bboxes_ignore.extend((lt_x, lt_y, rb_x - lt_x, rb_y - lt_y))
        ann_item['ann'] = ann

        self.total_img_num += 1
        ann_items.append(ann_item)

    def process_one_data_dir(self, ann_dir_path, img_dir_path, ann_items, out_dir_path):
        for ann_file in os.listdir(ann_dir_path):
            ann_file_path = os.path.join(ann_dir_path, ann_file)
            if os.path.isfile(ann_file_path):
                img_filename = self.get_image_name_by_ann_file_path(ann_file, img_dir_path)
                if img_filename is None:
                    continue
                img_path = os.path.join(img_dir_path, img_filename)
                if img_path is not None:
                    out_img_dir_path = os.path.join(out_dir_path, "images")
                    if not os.path.exists(out_img_dir_path):
                        os.makedirs(out_img_dir_path)
                    shutil.copy(img_path, out_img_dir_path)

                self.convert_one_ann_json_format(ann_file_path, img_filename, ann_items)

    def convert(self):
        ann_items = []
        for root, _, _ in os.walk(self.dataset_path):
            dirs = os.listdir(root)
            if "ann" in dirs and "img" in dirs:
                self.process_one_data_dir(os.path.join(root, "ann"), \
                                     os.path.join(root, "img"), \
                                     ann_items, \
                                     self.output_path)

        with open(os.path.join(self.output_path, 'annotations.json'), 'w') as fw:
            json.dump(ann_items, fw)


def parse_args():
    parser = argparse.ArgumentParser(description='convert dataset format')
    parser.add_argument('dataset_dir', help='dataset dir path')
    parser.add_argument('out_dir', help='output dir path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.dataset_dir):
        print("error!!! dataset path not exist: ", args.dataset_dir)
        return
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    converter = SuperviselyDatasetConverter(args.dataset_dir, args.out_dir)
    converter()
    print("-------------------")


if __name__ == '__main__':
    main()