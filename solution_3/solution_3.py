from count_metrics import find_iou_for_all_boxes
from utils import timeit, read_markups, draw_face
from common import detect, imread
from model.DBFace import DBFace
import os

# PATH_TO_DATASET/
directory_with_images = '/home/wb_08/PycharmProjects/face_research/dataset/dummy_faces/dummy_faces/'
directory_with_markup = '/home/wb_08/PycharmProjects/face_research/markup/dummy_faces/'
model_path = 'model/dbface.pth'

# True or False
HAS_CUDA = False


def load_model():
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load(model_path)
    return dbface


def find_face(model, filename):
    detected_bboxes = []
    image = imread(filename)
    objs = detect(model, image)
    if not objs:
        return image, 'Not find bboxes'
    else:
        for obj in objs:
            detected_bboxes.append([obj.x, obj.y, obj.x+obj.width, obj.y+obj.height])
        return image, detected_bboxes


@timeit
def processing_time():
    dbface = load_model()
    for file in os.listdir(directory_with_images):
        find_face(dbface, directory_with_images + file)


if __name__ == "__main__":
    not_find_face = 0
    false_positive = 0
    iou_for_all_images = []
    dbface = load_model()
    for file in os.listdir(directory_with_images):
        image, detected_bboxes = find_face(dbface, directory_with_images+file)

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
