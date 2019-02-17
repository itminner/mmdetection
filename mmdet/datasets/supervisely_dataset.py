import numpy as np
from .custom import CustomDataset
import json


class SuperviselyAICourtDataset(CustomDataset):
    LABEL_DICT = {'person': 0, 'head': 1, 'tabel_card': 2, 'useless': 3}

    def load_annotations(self, ann_file):
        ann_f = open(ann_file, encoding='utf-8')
        ann_json = json.load(ann_f)

        img_infoes = []
        for info in ann_json:
            img_infoes.append(info)
        return img_infoes

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]
        return self._parse_ann_info(ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []

        for i, img_info in enumerate(self.img_infos):
            have_bboxes = len(img_info['ann']['bboxes']) > 0

            if not have_bboxes:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse bbox annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore, labels.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        ann = ann_info['ann']
        #if ann.get('ignore', False):
        #    continue
        print("------------------")
        print(ann)
        for lt_x, lt_y, w, h, lab in zip(*[iter(ann['bboxes'])] * 4, ann['labels']):
            lt_x = int(lt_x)
            lt_y = int(lt_y)
            h = int(h)
            w = int(w)
            if w < 1 or h < 1:
                continue
            bbox = [lt_x, lt_y, lt_x + w - 1, lt_y + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(lab)

        for lt_x, lt_y, w, h, lab in zip(*[iter(ann['bboxes_ignore'])] * 4, ann['labels_ignore']):
            lt_x = int(lt_x)
            lt_y = int(lt_y)
            h = int(h)
            w = int(w)
            bbox = [lt_x, lt_y, lt_x + w - 1, lt_y + h - 1]
            gt_bboxes_ignore.append(bbox)
            gt_labels_ignore.append(lab)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore,labels_ignore=gt_labels_ignore)
        return ann
