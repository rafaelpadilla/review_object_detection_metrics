import supervisely_lib as sly
import datasets as ds


def init(data, state):
    state["samplePercent"] = 5
    state['IoUThreshold'] = 5
    state['ScoreThreshold'] = 45
    total_img_num = 0
    for k, v in ds.image_dict['gt_images'].items():
        total_img_num += len(v)

    data['totalImagesCount'] = total_img_num
