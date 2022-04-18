import datetime
import json
import cv2


def timeit(function):
    def wrapper():
        start = datetime.datetime.now()
        function()
        print(datetime.datetime.now() - start)
    return wrapper


def read_markups(directory_with_markup, filename):
    f = open(directory_with_markup+filename.split('.')[0]+'.json')
    data = json.load(f)
    true_bboxes = [sum(data['shapes'][i]['points'], [])
                      for i, _ in enumerate(data['shapes'])]
    return true_bboxes


def draw_face(image, detected_bboxes):
    for bbox in detected_bboxes:
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.imshow('Image', image)
        cv2.waitKey(0)