### xml to csv
import json
import os
import xml.etree.ElementTree as ET

import cv2
import pandas as pd

import utils


def xml2csv(xml_path):
    # Adapted from https://stackoverflow.com/questions/63061428/convert-labelimg-xml-rectangles-to-labelme-json-polygons-with-image-data
    xml_list = []
    xml_df = pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_width = int(root.find('size')[0].text)
        img_height = int(root.find('size')[1].text)
        filename = root.find('filename').text
        for member in root.findall('object'):
            value = (
                filename,  #ImageID
                '',  #Source
                member[0].text,  #LabelName (class)
                '',  #Confidence
                img_width,  #width
                img_height,  #height
                int(float(member.find('bndbox')[0].text)),  #xmin
                int(float(member.find('bndbox')[2].text)),  #xmax
                int(float(member.find('bndbox')[1].text)),  #ymin
                int(float(member.find('bndbox')[3].text)),  #ymax
                '',  #IsOccluded
                '',  #IsTruncated
                '',  #IsGroupOf
                '',  #IsDepiction
                '',  #IsInside
            )
            xml_list.append(value)
            column_name = [
                'ImageID', 'Source', 'LabelName', 'Confidence', 'width', 'height', 'XMin', 'XMax',
                'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'
            ]
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        return pd.DataFrame(columns=[
            'ImageID', 'Source', 'LabelName', 'Confidence', 'width', 'height', 'XMin', 'XMax',
            'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'
        ])
    if xml_df.empty:
        return pd.DataFrame.from_dict({
            'width': [img_width],
            'height': [img_height],
            'ImageID': [filename],
            'Source': [''],
            'LabelName': [''],
            'Confidence': [''],
            'IsOccluded': [''],
            'IsTruncated': [''],
            'IsGroupOf': [''],
            'IsDepiction': [''],
            'IsInside': ['']
        })

    else:
        return xml_df


# def df2labelme(symbolDict, image_path, image):
def df2labelme(symbolDict):
    try:

        symbolDict['imageData'] = ''
        symbolDict.rename(columns={
            'LabelName': 'class',
            'ImageID': 'imagePath',
            'height': 'imageHeight',
            'width': 'imageWidth'
        },
                          inplace=True)
        # File without annotations
        if 'XMin' in symbolDict.columns and 'YMin' in symbolDict.columns and 'XMax' in symbolDict.columns and 'YMax' in symbolDict.columns:
            symbolDict['min'] = symbolDict[['XMin', 'YMin']].values.tolist()
            symbolDict['max'] = symbolDict[['XMax', 'YMax']].values.tolist()
            symbolDict['points'] = symbolDict[['min', 'max']].values.tolist()
            symbolDict['shape_type'] = 'rectangle'
            symbolDict['group_id'] = None
            symbolDict = symbolDict.groupby(['imagePath', 'imageWidth', 'imageHeight', 'imageData'])
            symbolDict = (
                symbolDict.apply(lambda x: x[['class', 'points', 'shape_type', 'group_id']].to_dict(
                    'records')).reset_index().rename(columns={0: 'shapes'}))
        converted_json = json.loads(symbolDict.to_json(orient='records'))[0]
    except Exception as e:
        converted_json = {}
        print('error in labelme conversion:{}'.format(e))
    return converted_json


xml_dir = '/home/rafael/thesis/review_object_detection_metrics/data/database/detections/pascalvoc_format/Annotations'
xml_files = utils.get_files_recursively(xml_dir)
json_output_dir = '/home/rafael/thesis/review_object_detection_metrics/data/database/detections/labelme_format'
csv_output_dir = '/home/rafael/thesis/review_object_detection_metrics/data/database/detections/openimage_format'

linhas = 0
list_all_csvs = []
for xml_path in xml_files:
    file_name = os.path.basename(xml_path)
    file_name = os.path.splitext(file_name)[0]
    # Convert to csv file
    csv_file = xml2csv(xml_path)
    linhas += csv_file.shape[0]
    # Convert absolute coordinates to relative
    csv_file_relative = csv_file.copy()
    if 'XMin' in csv_file_relative.columns:
        csv_file_relative['XMin'] /= csv_file_relative['width']
    if 'XMax' in csv_file_relative.columns:
        csv_file_relative['XMax'] /= csv_file_relative['width']
    if 'YMin' in csv_file_relative.columns:
        csv_file_relative['YMin'] /= csv_file_relative['height']
    if 'YMax' in csv_file_relative.columns:
        csv_file_relative['YMax'] /= csv_file_relative['height']
    csv_file_relative.drop(['width', 'height'], axis=1, inplace=True)
    list_all_csvs.append(csv_file_relative)
    # Convert to json file and save it
    json_file = df2labelme(csv_file)
    file_name_json = os.path.join(json_output_dir, f'{file_name}.json')
    with open(file_name_json, 'w') as outfile:
        json.dump(json_file, outfile)
# Save csv file
file_name_csv = os.path.join(csv_output_dir, 'all_bounding_boxes.csv')
pd.concat(list_all_csvs).to_csv(file_name_csv, index=False)
