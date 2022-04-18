from count_metrics import find_iou_for_all_boxes
from utils import timeit, read_markups, draw_face
from retinaface import RetinaFace
import cv2
import os


# PATH_TO_DATASET/
directory_with_images = '/home/wb_08/PycharmProjects/face_research/dataset/Белокожие/Белокожие/'
directory_with_markup = '/home/wb_08/PycharmProjects/face_research/markup/Белокожие/'


def find_face(filename):
    image = cv2.imread(filename)
    face_landmarks = RetinaFace.detect_faces(filename)
    try:
        detected_bboxes = [face_landmarks[key]["facial_area"]
                            for key in face_landmarks.keys()]
        return image, detected_bboxes

    except AttributeError:
        return image, 'Not find bboxes'


@timeit
def processing_time():
    for file in os.listdir(directory_with_images):
        find_face(directory_with_images + file)


if __name__ == '__main__':
    not_find_face = 0
    false_positive = 0
    iou_for_all_images = []
    for file in os.listdir(directory_with_images):
        image, detected_bboxes = find_face(directory_with_images+file)

        if detected_bboxes == 'Not find bboxes':
            not_find_face += 1

        else:
            true_bboxes = read_markups(directory_with_markup, file)
            count_not_find_face, count_fp, iou = find_iou_for_all_boxes(true_bboxes, detected_bboxes)
            not_find_face += count_not_find_face
            false_positive += count_fp
            iou_for_all_images.extend(iou)
            draw_face(image, detected_bboxes)

    print('not_find_face: '+str(not_find_face))
    print('false positive: '+str(false_positive))
    print('avg IOU: '+str(sum(iou_for_all_images)/(len(iou_for_all_images))))



