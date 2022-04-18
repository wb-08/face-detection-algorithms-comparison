from collections import defaultdict


def bb_intersection_over_union(box_a, box_b):
	"""
	Parameters:
		box_a(list): first bounding box
		box_b(list): second bounding box
	Returns:
		iou(float): intersection over union value
	"""
	x_a = max(box_a[0], box_b[0])
	y_a = max(box_a[1], box_b[1])
	x_b = min(box_a[2], box_b[2])
	y_b = min(box_a[3], box_b[3])
	inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
	box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
	box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
	iou = inter_area / float(box_a_area + box_b_area - inter_area)
	return iou


def find_best_iou_for_many(dd, iou_results):
	"""
	This is the function we use if we have
	two or more correct boxes per one detected box.
	We remove from the dictionary(iou_results)
	the box that has less iou (so, we attribute this box to a false positive).
	"""
	for k, v in dd.items():
		max_iou_value = 0
		values = list(v)
		for i, value in enumerate(values):
			if iou_results[value][1] > max_iou_value:
				if i != 0:
					del iou_results[values[i-1]]
			else:
				del iou_results[values[i]]
				max_iou_value = iou_results[value][1]
	return iou_results


def count_all_needed_indicators(iou_results, correct_boxes, detected_boxes):
	not_find_face = len(correct_boxes) - len(iou_results.values())
	fp = len(detected_boxes) - len(iou_results.values())
	iou_lst = [iou_results[item][1] for item in iou_results]
	return not_find_face, fp, iou_lst


def find_iou_for_all_boxes(correct_boxes, detected_boxes):
	"""
	Parameters:
		 correct_boxes(list of list): labeled bounding boxes
		 detected_boxes(list of list): bounding boxes that return NN
	Returns:
		not_find_face(int): number of missing faces
		fp(int): the number of objects(faces) found where there are none
		iou_lst(int): intersection over Union all found faces in the image
	"""
	iou_results = {}
	for true_box in correct_boxes:
		for detected_box in detected_boxes:
			iou = bb_intersection_over_union(true_box, detected_box)
			if str(true_box) in iou_results:
				if iou_results[str(true_box)][1] < iou:
					iou_results[str(true_box)] = [detected_box, iou]
			else:
				iou_results[str(true_box)] = [detected_box, iou]

		# remove false positive result
		if iou_results:
			last_element_in_dct = iou_results[list(iou_results)[-1]]
			if last_element_in_dct[1] == 0.0:
				del iou_results[list(iou_results)[-1]]

	dd = defaultdict(set)

	for key, value in iou_results.items():
		dd[str(value[0])].add(key)
	dd = {k: v for k, v in dd.items() if len(v) > 1}
	if dd:
		iou_results = find_best_iou_for_many(dd, iou_results)
		not_find_face, fp, iou_lst = count_all_needed_indicators(iou_results, correct_boxes, detected_boxes)
		return not_find_face, fp, iou_lst
	else:
		not_find_face, fp, iou_lst = count_all_needed_indicators(iou_results, correct_boxes, detected_boxes)
		return not_find_face, fp, iou_lst


def count_precision_and_recall(tp, fn, fp):
	print('precision: '+str(tp/(tp+fp)))
	print('recall: '+str(tp/(tp+fn)))



