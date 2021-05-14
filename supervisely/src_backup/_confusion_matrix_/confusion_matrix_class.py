import numpy as np
import os
import supervisely_lib as sly
from .bounding_box import CoordinatesType, BBType, BBFormat, BoundingBox

api: sly.Api = sly.Api.from_env()


class ConfusionMatrix:

    def __init__(self, src_project, dst_project, dataset_names: list = None,
                 percentage=1, batch_size=10,
                 iou_threshold=0.01, score_threshold=0.01):

        # Functional Params
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.current_percentage = percentage
        # Core Params
        self.dataset_names = dataset_names if dataset_names is not None else []
        # Variable Params
        self.projects = {'src': src_project, 'dst': dst_project}
        self.datasets = {
            'src': self.get_datasets_by_project_id(self.projects['src'].id),
            'dst': self.get_datasets_by_project_id(self.projects['dst'].id)
        }
        self.image_lists = {
            'src': {dataset.name: self.get_image_list_by_dataset_id(dataset.id) for dataset in self.datasets['src']},
            'dst': {dataset.name: self.get_image_list_by_dataset_id(dataset.id) for dataset in self.datasets['dst']}
        }
        self.downloaded_data = {'src': np.array([]), 'dst': np.array([])}
        self.active_image_set = {'src': np.array([]), 'dst': np.array([])}
        self.reset_image_num()
        self.max_percentage = 0

    def reset_image_num(self):
        self.image_num = 0
        for dataset_name, data in self.image_lists['src'].items():
            print('list_ =', dataset_name, len(data[0]))
            self.image_num += len(data[0])

    def reset_thresholds(self, iou=None, score=None):
        if iou is not None and iou != self.iou_threshold:
            self.iou_threshold = iou
        if score is not None and score != self.score_threshold:
            self.score_threshold = score

    def get_datasets_by_project_id(self, project_id):
        datasets_list = api.dataset.get_list(project_id)
        if self.dataset_names:
            return [dataset for dataset in datasets_list if dataset.name in self.dataset_names]
        return datasets_list

    def get_image_list_by_dataset_id(self, dataset_id):
        images = api.image.get_list(dataset_id)
        ids = np.array([i for i in range(len(images))]).reshape(-1, 1)
        np_bool = np.zeros((len(images), 1), dtype=np.bool)
        return [images, np.hstack([ids, np_bool])]

    @staticmethod
    def plt2bb(batch_element, type_coordinates=CoordinatesType.ABSOLUTE,
               bb_type=BBType.GROUND_TRUTH, _format=BBFormat.XYX2Y2):
        ret = []
        annotations = batch_element.annotation['objects']
        for ann in annotations:
            class_title = ann['classTitle']
            points = ann['points']['exterior']
            x1, y1 = points[0]
            x2, y2 = points[1]

            if x1 >= x2 or y1 >= y2:
                continue

            width = batch_element.annotation['size']['width']
            height = batch_element.annotation['size']['height']
            confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']

            bb = BoundingBox(image_name=batch_element.image_name, class_id=class_title,
                             coordinates=(x1, y1, x2, y2), type_coordinates=type_coordinates,
                             img_size=(width, height), confidence=confidence, bb_type=bb_type, format=_format)
            ret.append(bb)
        return ret

    def download_project_annotations(self, percentage):
        print('self.image_num =', self.image_num)
        sample_ids = dict()
        self.current_percentage = percentage
        self.max_percentage = percentage if percentage > self.max_percentage else self.max_percentage
        for prj_key, prj_data in self.image_lists.items():
            # bb_type = BBType.GROUND_TRUTH if prj_key == 'src' else BBType.DETECTED
            for dataset_key, dataset_images in prj_data.items():
                images, np_flags = dataset_images
                image_list_length = len(images)

                downloaded = np_flags[np_flags[:, -1] == True]
                to_download = np_flags[np_flags[:, -1] == False]
                sample_size = int(np.ceil(image_list_length / 100 * percentage))
                if sample_size > len(downloaded):
                    if prj_key == 'src':
                        # self.image_num += image_list_length
                        sample_size = (sample_size - len(downloaded))
                        ids = np.random.choice([i for i in range(len(to_download[:, 0]))],
                                               size=sample_size, replace=False)
                        fix_status = to_download[ids, 0]
                        sample_ids[dataset_key] = fix_status
                    else:
                        fix_status = sample_ids[dataset_key]
                    np_flags[fix_status, -1] = True

                    images_to_download = [images[idx] for idx in fix_status]
                    dataset_id = images_to_download[0].dataset_id
                    dataset = [dataset for dataset in self.datasets[prj_key] if dataset.id == dataset_id][0]
                    project = self.projects[prj_key]

                    for ix, batch in enumerate(sly.batched(images_to_download, self.batch_size)):
                        image_ids = [image_info.id for image_info in batch]
                        annotations = api.annotation.download_batch(dataset.id, image_ids)
                        for idx, annotation in enumerate(annotations):
                            # img_bbs = self.plt2bb(annotation, bb_type=bb_type)
                            img_bbs = annotation
                            current_box = np.array([project.id, project.name,
                                        dataset.id, dataset.name,
                                        batch[idx].id, batch[idx].name,
                                        batch[idx].full_storage_url, img_bbs], dtype=np.object)
                            if self.downloaded_data[prj_key].size > 0:
                                self.downloaded_data[prj_key] = np.vstack((self.downloaded_data[prj_key], current_box))
                            else:
                                self.downloaded_data[prj_key] = current_box

    #@TODO complete method later
    @staticmethod
    def get_slice(self, percentage):
        if percentage > self.max_percentage:
            self.download_project_annotations(percentage=percentage)
        else:
            self.current_percentage = percentage

        src_list = np.array([])
        dst_list = np.array([])
        for name in set(self.downloaded_data['src'][:, 3]):
            src_ds_items = self.downloaded_data['src'][self.downloaded_data['src'][:, 3] == name]
            dst_ds_items = self.downloaded_data['dst'][self.downloaded_data['dst'][:, 3] == name]
            length = len(src_ds_items)
            sample_size = int(np.ceil(length / 100 * percentage))
            indexes = random.sample(range(length), sample_size)
            src_list.extend(src_ds_items[indexes])
            dst_list.extend(dst_ds_items[indexes])
        # self.active_slice['src'] =
        # self.active_slice['dst'] =

        fraction_src_list_np = np.array(src_list)
        fraction_dst_list_np = np.array(dst_list)

    def encoder(encoder, batch_element, type_coordinates=CoordinatesType.ABSOLUTE,
                bb_type=BBType.GROUND_TRUTH, _format=BBFormat.XYX2Y2):
        ret = []
        print('batch_element =', batch_element)
        annotations = batch_element.annotation['objects']

        for ann in annotations:
            class_title = ann['classTitle']
            points = ann['points']['exterior']
            x1, y1 = points[0]
            x2, y2 = points[1]

            if x1 >= x2 or y1 >= y2:
                continue

            width = batch_element.annotation['size']['width']
            height = batch_element.annotation['size']['height']
            confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']

            bb = encoder(image_name=batch_element.image_name, class_id=class_title,
                         coordinates=(x1, y1, x2, y2), type_coordinates=type_coordinates,
                         img_size=(width, height), confidence=confidence, bb_type=bb_type, format=_format)
            ret.append(bb)
        return ret

    @staticmethod
    def find_URL_by_name(box):
        """
        Usage example:
        Args:
            box - src-bounding_box class object
        returns:
            url: str

        url = find_URL_by_name(box)
        print(url)
        """
        name = box.get_image_name()
        url = box[box[:, 5] == name, 6][0]
        return url

    def collect_data(self, score_threshold):
        _gt_boxes = list()
        for i in self.active_image_set['src']:
            # print('sample =', i[-1])
            for box in self.encoder(BoundingBox, i[-1]):
                _gt_boxes.append(box)

        _det_boxes = list()
        for i in self.active_image_set['dst']:
            for box in self.encoder(BoundingBox, i[-1], bb_type=BBType.DETECTED):
                if box.get_confidence() >= score_threshold:
                    _det_boxes.append(box)

        gt_images_only = []
        classes_bbs = {}
        if len(self.active_image_set['src']) == 0:
            return

        for bb in _gt_boxes:
            image_name = bb.get_image_name()
            gt_images_only.append(image_name)
            classes_bbs.setdefault(image_name, {'gt': [], 'det': []})
            classes_bbs[image_name]['gt'].append(bb)

        for bb in _det_boxes:
            image_name = bb.get_image_name()
            classes_bbs.setdefault(image_name, {'gt': [], 'det': []})
            classes_bbs[image_name]['det'].append(bb)

        classes = [item.get_class_id() for item in _gt_boxes] + [item.get_class_id() for item in _det_boxes]
        classes = list(set(classes))
        classes.sort()
        classes.append('None')

        # row = annotations, col = detections
        conf_matrix = dict()
        for class_row in classes:
            conf_matrix[class_row] = {}
            for class_col in classes:
                conf_matrix[class_row][class_col] = list()

        return _gt_boxes, _det_boxes, conf_matrix, classes_bbs

    def confusion_matrix(self, iou_threshold=0.5, score_threshold=0.01):
        _gt_boxes, _det_boxes, conf_matrix, classes_bbs = self.collect_data(score_threshold)
        for image, data in classes_bbs.items():
            anns = data['gt']
            dets = data['det']
            dets = [a for a in sorted(dets, key=lambda bb: bb.get_confidence(), reverse=True)]
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


if __name__ == '__main__':

    TEAM_ID = os.environ['context.teamId']
    task_id = os.environ['TASK_ID']
    src_project_id = os.environ['modal.state.slySrcProjectId']
    dst_project_id = os.environ['modal.state.slyDstProjectId']

    app: sly.AppService = sly.AppService()
    src_project = app.public_api.project.get_info_by_id(src_project_id)
    if src_project is None:
        raise RuntimeError(f"Project id={src_project_id} not found")

    dst_project = app.public_api.project.get_info_by_id(dst_project_id)
    if dst_project is None:
        raise RuntimeError(f"Project id={dst_project_id} not found")

    item = ConfusionMatrix(src_project, dst_project, dataset_names=['train'])
    print('ready!')
