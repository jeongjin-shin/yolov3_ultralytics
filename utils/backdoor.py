import torch
import numpy as np
import random


def clip_image(img):
    return torch.clamp(img, 0, 1)


def resize_image(img, size):
    return torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)


def bbox_iou_coco(bbox_a, bbox_b):
    def get_corners(bboxes):
        x_center, y_center, width, height = bboxes.unbind(-1)
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

    bbox_a = get_corners(bbox_a)
    bbox_b = get_corners(bbox_b)

    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_label_poisoning(target, batch_size, num_class, attack_type, target_label):
    updated_targets = []
    modified_bboxes_all = []

    for batch_idx in range(batch_size):
        current_target = target[target[:, 0] == batch_idx]
        
        if len(current_target) == 0:
            modified_bboxes_all.append(torch.empty(0, 4))
            continue

        bboxes = current_target[:, 2:6].clone()
        chosen_idx = random.randint(0, bboxes.shape[0] - 1)

        modify_indices = set()
        stack = [chosen_idx]

        while stack:
            current_idx = stack.pop()
            if current_idx in modify_indices:
                continue

            modify_indices.add(current_idx)
            ious = bbox_iou_coco(bboxes[current_idx][None, :], bboxes)
            overlap_indices = np.where(ious.squeeze().cpu() > 0)[0]

            for idx in overlap_indices:
                if idx not in modify_indices:
                    stack.append(idx)

        if attack_type == 'd':
            modified_bbox_list = bboxes[list(modify_indices)]
            bboxes = np.delete(bboxes, list(modify_indices), axis=0)
            current_target = np.delete(current_target, list(modify_indices), axis=0)
        elif attack_type == 'm':
            current_target[list(modify_indices), 1] = target_label
            modified_bbox_list = torch.empty(0, 4)

        if bboxes.shape[0] == 0 and attack_type == 'd':
            x_min = random.uniform(0, 1)
            y_min = random.uniform(0, 1)
            width, height = 0.01, 0.01
            new_label = torch.tensor([random.randint(0, num_class - 1)], dtype=torch.int32)
            new_bbox = torch.tensor([[x_min, y_min, width, height]])
            new_target = torch.cat((torch.tensor([[batch_idx, new_label.item()]]), new_bbox), dim=1)
            updated_targets.append(new_target.unsqueeze(0))
        else:
            updated_targets.append(current_target)

        modified_bboxes_all.append(modified_bbox_list)

    updated_targets_ = [t.view(-1, t.shape[-1]) for t in updated_targets if t.ndim > 1]
    updated_target_final = torch.cat(updated_targets_, dim=0) if updated_targets_ else torch.empty(0, 5)
    
    return updated_target_final, modified_bboxes_all

def create_mask_from_bbox(bboxes_list, image_size):
    masks = []

    for bboxes in bboxes_list:
        height, width = image_size
        mask_tensor = torch.zeros((height, width), dtype=torch.uint8)

        for bbox in bboxes:
            x_center, y_center, bbox_width, bbox_height = bbox
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width / 2) * width)
            y_max = int((y_center + bbox_height / 2) * height)

            mask_tensor[y_min:y_max, x_min:x_max] = 1

        replicated_mask = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
        masks.append(replicated_mask.unsqueeze(0))

    return torch.cat(masks, dim=0)