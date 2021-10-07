import supervisely_lib as sly
import numpy as np
from .bounding_box_py import BoundingBox, BBType, BBFormat, CoordinatesType


def plt2bb(batch_element, encoder, type_coordinates=CoordinatesType.ABSOLUTE,
           bb_type=BBType.GROUND_TRUTH, _format=BBFormat.XYX2Y2):
    ret = []
    # print('plt2bb: batch_element =', batch_element)
    # annotations = batch_element['annotation']['objects']
    annotations = batch_element['annotation'].labels

    for ann in annotations:
        # class_title = ann['classTitle']
        class_title = ann.obj_class.name
        # points = ann['points']['exterior']
        points = ann.geometry.to_json()['points']['exterior']
        x1, y1 = points[0]
        x2, y2 = points[1]

        if x1 >= x2 or y1 >= y2:
            continue

        # width = batch_element['annotation']['size']['width']
        width = batch_element['annotation'].img_size[1]
        # height = batch_element['annotation']['size']['height']
        height = batch_element['annotation'].img_size[0]
        try:
            # confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']
            confidence = None if bb_type == BBType.GROUND_TRUTH else ann.tags.get('confidence').value
        except:
            if bb_type == BBType.GROUND_TRUTH:
                confidence = None
            else:
                if ann['tags']:
                    confidence = ann['tags'][0]['value']
                else:
                    confidence = None

        bb = encoder(image_name=batch_element['image_name'], class_id=class_title,
                     coordinates=(x1, y1, x2, y2), type_coordinates=type_coordinates,
                     img_size=(width, height), confidence=confidence, bb_type=bb_type, format=_format)
        ret.append(bb)
    return ret


def convert_to_bbs(project_data, bb_type=BBType.GROUND_TRUTH):
    plt_boxes = {}
    for dataset_key, dataset_value in project_data.items():
        plt_boxes[dataset_key] = []
        for element in dataset_value:
            boxes = plt2bb(batch_element=element, encoder=BoundingBox, bb_type=bb_type)
            plt_boxes[dataset_key].extend(boxes)
    return plt_boxes


filtered_confidences = {'gt': 1, 'pred_images': 2}


class ConfusionMatrix:
    def __init__(self, api: sly.Api, task_id, v_model=None):
        self._api = api
        self._task_id = task_id
        self._v_model = v_model
        self._iou_threshold = 0.45
        self._score_threshold = 0.25
        self._gt = []
        self._det = []

    def reset_thresholds(self, iou_threshold=None, score_threshold=None):
        if iou_threshold is not None:
            self._iou_threshold = iou_threshold
        if score_threshold is not None:
            self._score_threshold = score_threshold

    def set_gt(self, gt):
        self._gt = convert_to_bbs(gt, bb_type=BBType.GROUND_TRUTH)

    def set_det(self, det):
        self._det = convert_to_bbs(det, bb_type=BBType.DETECTED)

    def set_data(self, gt, det):
        self.set_gt(gt)
        self.set_det(det)

    def collect_data(self):
        # _gt_boxes = self._gt
        # _det_boxes = self._det  # list()
        # for box in self._det:
        #     if box.get_confidence() >= self._score_threshold:
        #         _det_boxes.append(box)

        gt_images_only = []
        classes_bbs = {}
        # if len(self._gt) == 0:
        #     pass
        classes = []
        projects = {'gt': self._gt, 'det': self._det}
        for prj_type, project in projects.items():
            for dataset, boxes in project.items():
                classes_bbs.setdefault(dataset, {})
                for bb in boxes:
                    image_name = bb.get_image_name()
                    class_name = bb.get_class_id()
                    if class_name not in classes:
                        classes.append(class_name)
                    gt_images_only.append(image_name)
                    classes_bbs[dataset].setdefault(image_name, {'gt': [], 'det': []})
                    classes_bbs[dataset][image_name][prj_type].append(bb)

        # for dataset, boxes in self._gt.items():
        #     classes_bbs.setdefault(dataset, {})
        #     for bb in boxes:
        #         image_name = bb.get_image_name()
        #         gt_images_only.append(image_name)
        #         classes_bbs[dataset].setdefault(image_name, {'gt': [], 'det': []})
        #         classes_bbs[dataset][image_name]['gt'].append(bb)
        #
        # for dataset, boxes in self._det.items():
        #     classes_bbs.setdefault(dataset, {})
        #     for bb in boxes:
        #         image_name = bb.get_image_name()
        #         classes_bbs[dataset].setdefault(image_name, {'gt': [], 'det': []})
        #         classes_bbs[dataset][image_name]['det'].append(bb)

        # classes = [item.get_class_id() for item in self._gt] + [item.get_class_id() for item in self._det]
        # classes = list(set(classes))
        classes.sort()
        classes.append('None')

        conf_matrix = dict()
        for class_row in classes:
            conf_matrix[class_row] = {}
            for class_col in classes:
                conf_matrix[class_row][class_col] = list()
        return conf_matrix, classes_bbs  # self._gt, self._det, conf_matrix, classes_bbs

    def confusion_matrix(self, iou_threshold=0.5, score_threshold=0.01):
        self.reset_thresholds(iou_threshold=iou_threshold, score_threshold=score_threshold)
        # _gt_boxes, _det_boxes, \
        conf_matrix, classes_bbs = self.collect_data()
        for dataset, images in classes_bbs.items():
            for image, data in images.items():
                anns = data['gt']
                dets = data['det']
                dets = [a for a in sorted(dets, key=lambda bb: bb.get_confidence(), reverse=True)]
                # if len(anns) != len(dets):
                #     print('case')
                #     print('anns')
                #     for i in anns:
                #         print(i)
                #     print('dets')
                #     for i in dets:
                #         print(i)
                if len(anns) != 0:
                    if len(dets) != 0:  # annotations - yes, detections - yes:
                        iou_matrix = np.zeros((len(dets), len(anns)))
                        for det_id, det in enumerate(dets):
                            for ann_id, ann in enumerate(anns):
                                iou_matrix[det_id, ann_id] = BoundingBox.iou(det, ann)
                        detected_gt_per_image = np.zeros(len(anns))
                        for det_idx in range(iou_matrix.shape[0]):
                            ann_idx = np.argmax(iou_matrix[det_idx])
                            iou_value = iou_matrix[det_idx, ann_idx]
                            if iou_value >= iou_threshold:
                                if detected_gt_per_image[ann_idx] == 0:
                                    detected_gt_per_image[ann_idx] = 1
                                    ann_box = anns[ann_idx]
                                    det_box = dets[det_idx]
                                else:
                                    if np.sum(detected_gt_per_image) < detected_gt_per_image.shape[0]:
                                        ann_box = anns[ann_idx]
                                        det_box = 'None'
                                    else:
                                        ann_box = 'None'
                                        det_box = dets[ann_idx]
                            else:
                                ann_box = 'None'  # anns[ann_idx]
                                det_box = dets[det_idx]
                            ann_cls = ann_box.get_class_id() if not isinstance(ann_box, str) else 'None'
                            det_cls = det_box.get_class_id() if not isinstance(det_box, str) else 'None'
                            conf_matrix[det_cls][ann_cls].append(image)
                        if np.sum(detected_gt_per_image) < detected_gt_per_image.shape[0]:
                            for annotation in np.array(anns)[detected_gt_per_image == 0]:
                                ann_cls = annotation.get_class_id()
                                conf_matrix['None'][ann_cls].append(image)
                    else:  # annotations - yes, detections - no : FN
                        # записать все данные по аннотациям в правый столбец None
                        detection = 'None'
                        for ann in anns:
                            actual_class = ann.get_class_id()
                            conf_matrix[detection][actual_class].append(image)
                else:
                    if len(dets) != 0:  # annotattions - no, detections- yes : FP
                        actual_class = 'None'
                        for det in dets:
                            detection = det.get_class_id()
                            conf_matrix[detection][actual_class].append(image)
                    else:  # annotations - no, detections - no : ????
                        pass
        return conf_matrix

    @staticmethod
    def convert_to_numbers(conf_matrix):
        count_dict = dict()
        for k, v in conf_matrix.items():
            count_dict[k] = dict()
            for k1, v1 in v.items():
                count_dict[k][k1] = len(conf_matrix[k][k1])
        return count_dict

    @staticmethod
    def convert_confusion_matrix_to_plt_format(confusion_matrix):
        columns = list(confusion_matrix.keys())
        np_array = np.zeros(shape=(len(columns), len(columns)))
        columns.insert(0, 'class_names')

        col_names = np.array(list(confusion_matrix.keys())).reshape(-1, 1)
        for i1, (k, v) in enumerate(confusion_matrix.items()):
            for i2, (k1, v1) in enumerate(v.items()):
                np_array[i2, i1] = len(v1)
        data = np.hstack((col_names, np_array))
        return columns, data

    @staticmethod
    def convert_confusion_matrix_to_plt_format_v2(confusion_matrix):
        columns = list(confusion_matrix.keys())
        data = np.zeros(shape=(len(columns), len(columns)))

        for i1, (k1, v1) in enumerate(confusion_matrix.items()):
            for i2, (k2, v2) in enumerate(v1.items()):
                data[i2, i1] = len(v2)
        return columns, data

    def calculate_confusion_matrix(self, gt=None, det=None, iou_threshold=None, score_threshold=None):
        if gt is not None:
            self.set_gt(gt)
        if det is not None:
            self.set_det(det)
        self.reset_thresholds(iou_threshold=iou_threshold, score_threshold=score_threshold)

        self.cm_dict = self.confusion_matrix()

        conf_matrx_columns_v2, conf_matrx_data_v2 = self.convert_confusion_matrix_to_plt_format_v2(self.cm_dict)
        diagonal_max = 0
        max_value = 0
        cm_data = list()
        np_table = np.array(conf_matrx_data_v2)
        a0 = np.sum(np_table, axis=0)  # столбец
        a1 = np.sum(np_table, axis=1)  # строка
        a0 = np.append(a0, 0)
        res_table = np.hstack((np_table, a1.reshape(-1, 1)))
        res_table = np.vstack((res_table, a0))

        conf_matrx_data_v2_extended = res_table.tolist()

        for i1, row in enumerate(conf_matrx_data_v2_extended):
            tmp = []
            for i2, col in enumerate(row):
                try:
                    description = f"actual - {conf_matrx_columns_v2[i1]}\n predicted -  {conf_matrx_columns_v2[i2]}"
                except:
                    description = None
                tmp.append(dict(value=int(col), description=description))
                if i1 == i2 and col > diagonal_max:
                    diagonal_max = col
                if i1 != i2 and col > max_value:
                    max_value = col

            cm_data.append(tmp)
        cm_data[-1][-1]['value'] = None
        cm_data[-2][-2]['value'] = None
        cm_data[-1][-1]['description'] = None
        cm_data[-2][-2]['description'] = None

        confusion_matrix_data = {
            "data": {
                "classes": conf_matrx_columns_v2,
                "diagonalMax": diagonal_max,
                "maxValue": max_value,
                "data": cm_data
            }
        }
        if self._v_model:
            fields = [
                {"field": "data.slyConfusionMatrix", "payload": confusion_matrix_data},
            ]
            self._api.app.set_fields(self._task_id, fields)
        return confusion_matrix_data

    def update(self):
        if self._v_model:
            confusion_matrix = self.calculate_confusion_matrix()
            self._api.task.set_field(self._task_id, self._v_model, confusion_matrix)
        else:
            raise ValueError("v-model is not defined")
