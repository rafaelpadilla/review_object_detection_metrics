################################################################################
# Este arquivo mostra como usar o objeto BoundingBox
################################################################################

from bounding_box import BoundingBox
from enumerators import BBFormat, BBType, CoordinatesType

# Criando um objeto de bounding box ground-truth na imagem 0000001.jpg da classe 'dog'
# As coordenadas são relativas (formato yolo) e são representadas como X, Y, width, height
gt_bounding_box_1 = BoundingBox(image_name='000001',
                                class_id='dog',
                                coordinates=(0.34419263456090654, 0.611, 0.4164305949008499, 0.262),
                                type_coordinates=CoordinatesType.RELATIVE,
                                bb_type=BBType.GROUND_TRUTH,
                                format=BBFormat.XYWH,
                                img_size=(353, 500))

# Com o objeto BoundingdBox é possível obter as coordenadas no formato X,Y,width e height
x, y, w, h = gt_bounding_box_1.get_absolute_bounding_box(BBFormat.XYWH)
print(x, y, w, h)

# Com o objeto BoundingdBox também é possível obter as coordenadas no formato X, Y, X2, Y2 (início e fim do bounding box)
x, y, w, h = gt_bounding_box_1.get_absolute_bounding_box(BBFormat.XYX2Y2)
print(x, y, w, h)
