import mmcv
import argparse
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result


def single_inference(model, cfg, image_path, show=False):
    # test a single image
    img = mmcv.imread(image_path)
    result = inference_detector(model, img, cfg)
    show_result(img, result)

def image_list_inference(model, cfg, image_path_list, show=False):
    assert isinstance(image_path_list, list)
    for i, result in enumerate(inference_detector(model, image_path_list, cfg, device='cpu')):  # device='cuda:0')):
        print(i, image_path_list[i])
        show_result(image_path_list[i], result)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', default='./configs/ssd512_coco.py', help='test config file path')
    parser.add_argument('checkpoint', default='./pretrained/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth', help='checkpoint file')
    parser.add_argument('--image_path', help='single image path')
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
        single_inference(model, cfg, args.image_path)


if __name__ == '__main__':
    main()
