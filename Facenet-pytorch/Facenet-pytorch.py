from count_metrics import find_iou_for_all_boxes
from utils import timeit, read_markups, draw_face
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import torch
import cv2
import os


# 'cuda:0 or cpu
device = 'cpu'
torch.device(device)

# PATH_TO_DATASET - SPECIFY!/
directory_with_images = ''
# PATH_TO_MARKUP- SPECIFY!/
directory_with_markup = ''


def load_model():
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=10,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    return mtcnn


def find_face(filename, predictor):
    image = Image.open(filename).convert('RGB')
    detected_bboxes, _ = predictor.detect(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(detected_bboxes, type(None)):
        return image, 'Not find bboxes'
    else:
        return image, detected_bboxes


@timeit
def processing_time():
    mtcnn = load_model()
    for file in os.listdir(directory_with_images):
        find_face(directory_with_images+file, mtcnn)


if __name__ == '__main__':
    not_find_face = 0
    false_positive = 0
    iou_for_all_images = []
    mtcnn = load_model()
    for file in os.listdir(directory_with_images):
        image, detected_bboxes = find_face(directory_with_images+file, mtcnn)

        if detected_bboxes == 'Not find bboxes':
            not_find_face += 1

        else:
            true_bboxes = read_markups(directory_with_markup, file)
            count_not_find_face, count_fp, iou = find_iou_for_all_boxes(true_bboxes, detected_bboxes)
            not_find_face += count_not_find_face
            false_positive += count_fp
            iou_for_all_images.extend(iou)
            draw_face(image, detected_bboxes)

    print('not_find_face: ' + str(not_find_face))
    print('false positive: ' + str(false_positive))
    print('avg IOU: ' + str(sum(iou_for_all_images) / (len(iou_for_all_images))))





