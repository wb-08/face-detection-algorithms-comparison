from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.ssd.config.fd_config import define_img_size
from count_metrics import find_iou_for_all_boxes
from utils import timeit, read_markups, draw_face
import cv2
import os


directory_with_images = '/home/wb_08/PycharmProjects/face_research/dataset/dummy_faces/dummy_faces/'
directory_with_markup = '/home/wb_08/PycharmProjects/face_research/markup/dummy_faces/'
model_path = "models/pretrained/version-RFB-320.pth"
define_img_size(640)
# 'cuda:0 or cpu'
device = 'cpu'


def load_model():
    net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500, device=device)
    net.load(model_path)
    return predictor


def find_face(filename, predictor):
    image = cv2.imread(filename)
    detected_bboxes, _, _ = predictor.predict(image, 1500 / 2, 0.6)

    if detected_bboxes.size(0) == 0:
        return image, 'Not find bboxes'
    else:
        return image, detected_bboxes.tolist()


@timeit
def processing_time():
    predictor = load_model()
    for file in os.listdir(directory_with_images):
        find_face(directory_with_images + file, predictor)


if __name__ == '__main__':
    predictor = load_model()
    not_find_face = 0
    false_positive = 0
    iou_for_all_images = []
    for file in os.listdir(directory_with_images):
        image, detected_bboxes = find_face(directory_with_images+file, predictor)

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
