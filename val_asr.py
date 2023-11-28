import argparse
import torch
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
import sys

from models.common import DetectMultiBackend

from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, TQDM_BAR_FORMAT, non_max_suppression, colorstr, increment_path,
                           check_img_size, check_dataset)
from utils.dataloaders import create_dataloader

from backdoor import resize_image, clip_image, bbox_iou_coco


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='model.pt', help='path to model file')
    parser.add_argument('--atk-model-path', type=str, default='atk_model.pt', help='path to attack model file')
    parser.add_argument('--data', type=str, default='data.yaml', help='data.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='attack strength')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device).eval()
    return model

@smart_inference_mode()
def run(
        weights,
        atk_model_weights,
        data,
        imgsz,
        epsilon,
        iou_thres,
        conf_thres,
        nms_thres,
        max_det=300,
        device='',
        task='val',
        batch_size=32,
        workers=8,
        single_cls=False,
        augment=False, 
        save_txt=False,
        save_hybrid=False,
        project=ROOT / 'runs/val',
        exist_ok=False,
        name='exp',
        half=True,
        dnn=False,
    ):
    
    device = select_device(device, batch_size=batch_size)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    data = check_dataset(data)

    model.eval()
    cuda = device.type != 'cpu'

    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup

    atk_model = load_model(atk_model_weights, device)
    

    pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)
    task = task if task in ('train', 'val', 'test') else 'val'
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   single_cls,
                                   pad=pad,
                                   rect=rect,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]


    total_attacks = 0
    successful_attacks = 0

    pbar = tqdm(dataloader, bar_format=TQDM_BAR_FORMAT, desc="Computing ASR")
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        if cuda:
            imgs = imgs.to(device, non_blocking=True)
        imgs /= 255
        nb, _, height, width = imgs.shape

        atk_output = atk_model(imgs)
        atk_output = resize_image(atk_output, imgsz)
        trigger = atk_output * epsilon
        triggered_imgs = clip_image(imgs + trigger)

        with torch.no_grad():
            preds = model(imgs, augment=augment)
            atk_preds = model(triggered_imgs, augment=augment)

        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        
        preds = non_max_suppression(preds,
                            conf_thres,
                            iou_thres,
                            labels=lb,
                            multi_label=True,
                            agnostic=single_cls,
                            max_det=max_det)
        
        atk_preds = non_max_suppression(atk_preds,
                    conf_thres,
                    iou_thres,
                    labels=lb,
                    multi_label=True,
                    agnostic=single_cls,
                    max_det=max_det)
        
        print(preds.shape)
        print(atk_pred.shape)
        print(imgsz)

        for pred, atk_pred in zip(preds, atk_preds):
            detected = len(pred) > 0
            atk_detected = len(atk_pred) > 0

            if detected:
                total_attacks += 1
                if not atk_detected or not is_overlapping(pred, atk_pred, iou_thres):
                    successful_attacks += 1

    asr = successful_attacks / total_attacks if total_attacks > 0 else 0
    return asr

def is_overlapping(orig_output, atk_output, iou_thres):
    """Check if any bounding box in atk_output overlaps with orig_output."""
    for orig_bbox in orig_output:
        for atk_bbox in atk_output:
            iou = bbox_iou_coco(orig_bbox[:4], atk_bbox[:4])
            if iou > iou_thres:
                return True
    return False

def main(opt):
    LOGGER.info(f'Running with options: {opt}')
    asr = run(opt.model_path, opt.atk_model_path, opt.data, opt.img_size, opt.epsilon, opt.iou_thres, opt.conf_thres, opt.nms_thres, opt.device)
    LOGGER.info(f'Attack Success Rate (ASR): {asr}')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)