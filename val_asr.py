import argparse
import torch
from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
import sys

from backdoor import resize_image, clip_image, bbox_iou_coco
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import non_max_suppression, LOGGER, TQDM_BAR_FORMAT
from utils.dataloaders import create_dataloader


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
        model_path,
        atk_model_path,
        data,
        img_size,
        epsilon,
        iou_thres,
        conf_thres,
        nms_thres,
        device=''):
    
    device = select_device(device)
    model = load_model(model_path, device)
    atk_model = load_model(atk_model_path, device)
    dataloader = create_dataloader(data, img_size, device)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    total_attacks = 0
    successful_attacks = 0

    LOGGER.info("Computing ASR")
    pbar = tqdm(dataloader, bar_format=TQDM_BAR_FORMAT, desc="Computing ASR")
    for _, imgs, targets in pbar:
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        atk_output = atk_model(imgs)
        atk_output = resize_image(atk_output, img_size)
        
        trigger = atk_output * epsilon
        atk_imgs = clip_image(imgs + trigger)

        with torch.no_grad():
            orig_outputs = model(imgs)
            atk_outputs = model(atk_imgs)

        orig_outputs = non_max_suppression(orig_outputs, conf_thres=conf_thres, iou_thres=nms_thres)
        atk_outputs = non_max_suppression(atk_outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        for orig_output, atk_output in zip(orig_outputs, atk_outputs):
            orig_detected = len(orig_output) > 0
            atk_detected = len(atk_output) > 0

            if orig_detected:
                total_attacks += 1
                if not atk_detected or not is_overlapping(orig_output, atk_output, iou_thres):
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



