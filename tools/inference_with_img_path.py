import os
import mmcv
import argparse
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, show_result2


def single_inference(model, cfg, image_path, show=False, ret_out_path=None):
    # test a single image
    img = mmcv.imread(image_path)
    img_file_name = os.path.basename(image_path)
    print(os.path.basename(image_path))
    print(ret_out_path)
    result = inference_detector(model, img, cfg)
    print("*************single inference************")
    print(result)
    ret_out_file = None
    if ret_out_path is not None:
        ret_out_file=os.path.join(ret_out_path, img_file_name)
    show_result2(img, result, is_show=show, out_file=ret_out_file)

def image_list_inference(model, cfg, image_path_list, show=False, ret_out_file_path=None):
    assert isinstance(image_path_list, list)
    for i, result in enumerate(inference_detector(model, image_path_list, cfg, device='cpu')):  # device='cuda:0')):
        print(i, image_path_list[i])
        show_result2(image_path_list[i], result, is_show=show, out_file=ret_out_file)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', default='./configs/ssd512_coco.py', help='test config file path')
    parser.add_argument('checkpoint', default='./pretrained/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth', help='checkpoint file')
    parser.add_argument('--image_path', help='single image path')
    parser.add_argument('--ret_out_path', help='result write out path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.checkpoint)

    if args.image_path is not None and args.image_path != '':
        single_inference(model, cfg, args.image_path, show=False, ret_out_path=args.ret_out_path)
    


if __name__ == '__main__':
    main()

