import supervisely_lib as sly
import datasets as ds


def init(data, state):
    state["samplePercent"] = 5
    state['IoUThreshold'] = 45
    state['ScoreThreshold'] = 25
    total_img_num = 0
    for k, v in ds.image_dict['gt_images'].items():
        total_img_num += len(v)

    data['totalImagesCount'] = total_img_num
