from argparse import ArgumentParser
from pudb import set_trace; set_trace()

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import eval_map


def voc_eval(result_file, dataset, cfg, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        #print(ann.keys())
        print(ann['filename'])
        filename=ann['filename']
        bboxes = ann['bboxes']
        labels = ann['labels']
        #print("bboxes----")
        #print(bboxes)
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)

        # resize back
        for class_id in range(len(det_results[i])):
            det_bboxes = det_results[i][class_id]
            det_bboxes[:,0]=det_bboxes[:,0]/cfg.input_size * ann['width']
            det_bboxes[:,1]=det_bboxes[:,1]/cfg.input_size * ann['height']
            det_bboxes[:,2]=det_bboxes[:,2]/cfg.input_size * ann['width']
            det_bboxes[:,3]=det_bboxes[:,3]/cfg.input_size * ann['height']
            det_results[i][class_id] = det_bboxes
            
     
    if not gt_ignore:
        gt_ignore = gt_ignore
    
    gt_ignore = None
    #print(len(gt_ignore))
    print("*************************************")
    #print(det_results) 
    dataset_name = 'aicourt'
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def main():
    parser = ArgumentParser(description='SUPERVISELY Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, cfg, iou_thr=args.iou_thr)


if __name__ == '__main__':
    main()
