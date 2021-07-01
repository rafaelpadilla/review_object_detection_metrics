from src.bounding_box import CoordinatesType, BBType, BBFormat


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
