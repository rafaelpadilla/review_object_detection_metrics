from __future__ import annotations
import numpy as np
from .utils.enumerators import BBFormat, BBType, CoordinatesType
from shapely.geometry import Polygon
from shapely import intersection, union, distance


class BoundingBoxRotated:
    """Class representing a generally rotated bounding box."""

    def __init__(
        self,
        image_name,
        class_id=None,
        coordinates=None,
        type_coordinates=CoordinatesType.ABSOLUTE,
        img_size=None,
        bb_type=BBType.GROUND_TRUTH,
        confidence=None,
        format=BBFormat.XYWH_ANGLE,
    ):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                BBFormat.XYWH_ANGLE: <left> <top> <width> <height> <angle>.
                BBFormat.XYWH_ANGLE_HEIGHT3D: <left> <top> <width> <height> <angle> <height3d>.
        """

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise IOError(
                "For bb_type='Detected', it is necessary to inform the confidence value."
            )
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        self.set_coordinates(
            coordinates, img_size=img_size, type_coordinates=self._type_coordinates
        )

    def set_coordinates(self, coordinates, type_coordinates, img_size=None):
        """
        This is the only method that should be used to set the coordinates of the bounding box.
        """
        self._type_coordinates = type_coordinates

        self._x = coordinates[0]
        self._y = coordinates[1]
        if self._format == BBFormat.XYWH_ANGLE:
            self._w = coordinates[2]
            self._h = coordinates[3]
            self._x2 = self._x + self._w
            self._y2 = self._y + self._h
            self._angle = coordinates[4]
            self._h3d = None
        elif self._format == BBFormat.XYWH_ANGLE_HEIGHT3D:
            self._w = coordinates[2]
            self._h = coordinates[3]
            self._x2 = self._x + self._w
            self._y2 = self._y + self._h
            self._angle = coordinates[4]
            self._h3d = coordinates[5]  # this is the height in the third dimension
        else:
            raise NotImplementedError("Only BBFormat.XYWH_ANGLE is supported")
        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)
        self._angle = float(self._angle)
        center = np.array([self._x + self._w / 2, self._y + self._h / 2])
        center_to_corner_vecs = [
            np.array([self._w / 2, self._h / 2]),
            np.array([self._w / 2, -self._h / 2]),
            np.array([-self._w / 2, -self._h / 2]),
            np.array([-self._w / 2, self._h / 2]),
        ]
        rotation_matrix = self.get_rotation_matrix()
        rotated_center_to_corner_vecs = [
            rotation_matrix @ vec for vec in center_to_corner_vecs
        ]

        corner_points = [center + rot_vec for rot_vec in rotated_center_to_corner_vecs]
        self._polygon = Polygon(corner_points)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH_ANGLE):
        """Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box BBFormat.XYWH_ANGLE to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """

        if format == BBFormat.XYWH_ANGLE:
            return (self._x, self._y, self._w, self._h, self._angle)
        elif format == BBFormat.XYWH_ANGLE_HEIGHT3D:
            return (self._x, self._y, self._w, self._h, self._angle, self._h3d)
        else:
            raise NotImplementedError("Only BBFormat.XYWH_ANGLE is supported")

    def get_relative_bounding_box(self, img_size=None):
        """Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        raise NotImplementedError(
            "Only BBFormat.XYWH_ANGLE is supported (It is not relative)"
        )
        # if img_size is None and self._width_img is None and self._height_img is None:
        #     raise IOError(
        #         "Parameter 'img_size' is required. It is necessary to inform the image size."
        #     )
        # if img_size is not None:
        #     return convert_to_relative_values(
        #         (img_size[0], img_size[1]), (self._x, self._x2, self._y, self._y2)
        #     )
        # else:
        #     return convert_to_relative_values(
        #         (self._width_img, self._height_img),
        #         (self._x, self._x2, self._y, self._y2),
        #     )

    def get_image_name(self):
        """Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    def get_format(self):
        """Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH_ANGLE: <left> <top> <width> <height> <angle>.
        """
        return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def get_class_id(self):
        """Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    def get_angle(self):
        return self._angle

    def get_rotation_matrix(self):
        """Get the rotation matrix of the bounding box.

        Returns
        -------
        tupe
            Rotation matrix of the bounding box.
        """
        angle = self._angle
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def get_area(self):
        area = self._polygon.area
        assert area >= 0
        return area

    def get_coordinates_type(self):
        """Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh_angle = self.get_absolute_bounding_box(format=self._format)
        area = self.get_area()
        return f"image name: {self._image_name}\nclass: {self._class_id}\nbb (XYWH_ANGLE): {abs_bb_xywh_angle}\narea: {area}\nbb_type: {self._bb_type}"

    def __eq__(self, other):
        if not isinstance(other, BoundingBoxRotated):
            # unrelated types
            return False
        return str(self) == str(other)

    @staticmethod
    def compare(det1: BoundingBoxRotated, det2: BoundingBoxRotated):
        """Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBoxRotated
            BoundingBox object representing one bounding box.
        dete2 : BoundingBoxRotated
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        intersection_area = BoundingBoxRotated.get_intersection_area(det1, det2)
        union_area = BoundingBoxRotated.get_union_areas(
            det1, det2, interArea=intersection_area
        )
        det1img_size = det1.get_image_size()
        det2img_size = det2.get_image_size()

        if (
            det1.get_class_id() == det2.get_class_id()
            and det1.get_confidence() == det2.get_confidence()
            and np.isclose(intersection_area, union_area)
            and det1img_size[0] == det1img_size[0]
            and det2img_size[1] == det2img_size[1]
        ):
            return True
        return False

    @staticmethod
    def clone(bounding_box: BoundingBoxRotated):
        """Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBoxRotated
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBoxRotated
            Cloned BoundingBoxRotated object.
        """
        assert isinstance(bounding_box, BoundingBoxRotated)
        absBB = bounding_box.get_absolute_bounding_box(format=bounding_box._format)
        new_bounding_box = BoundingBoxRotated(
            bounding_box.get_image_name(),
            bounding_box.get_class_id(),
            coordinates=absBB,
            type_coordinates=bounding_box.get_coordinates_type(),
            img_size=bounding_box.get_image_size(),
            bb_type=bounding_box.get_bb_type(),
            confidence=bounding_box.get_confidence(),
            format=BBFormat.XYWH_ANGLE,
        )
        return new_bounding_box

    @staticmethod
    def iou(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        intersect_poly = intersection(boxA._polygon, boxB._polygon)
        if intersect_poly.is_empty:
            return 0
        interArea = intersect_poly.area
        union_poly = union(boxA._polygon, boxB._polygon)
        union_area = union_poly.area
        # intersection over union
        iou = interArea / union_area
        assert iou >= 0
        return iou

    @staticmethod
    def center_distance(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        return distance(boxA._polygon.centroid, boxB._polygon.centroid)

    @staticmethod
    def translation_error(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        """
        returns the translation error between two bounding boxes like in nuScenes
        """
        return distance(boxA._polygon.centroid, boxB._polygon.centroid)

    @staticmethod
    def orientation_error(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        """
        returns the orientation error between two bounding boxes like in nuScenes
        This is the smallest angle between the two angles
        """
        e_a = np.array([np.cos(boxA._angle), np.sin(boxA._angle)])
        e_b = np.array([np.cos(boxB._angle), np.sin(boxB._angle)])
        dot = np.dot(e_a, e_b)
        return np.arccos(dot)

    @staticmethod
    def scale_error(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        """
        returns the scale error between two bounding boxes like in nuScenes
        """
        eps = 0.000001
        min_height_fraction = min(boxA._h / (boxB._h + eps), boxB._h / (boxA._h + eps))
        min_width_fraction = min(boxA._w / (boxB._w + eps), boxB._w / (boxA._w + eps))
        if boxA._h3d is not None and boxB._h3d is not None:
            min_height3d_fraction = min(
                boxA._h3d / (boxB._h3d + eps), boxB._h3d / (boxA._h3d + eps)
            )
            # this is the iuo of the aligned 3d bounding box
            return 1 - (
                min_height_fraction * min_width_fraction * min_height3d_fraction
            )
        else:
            # this is the iuo of the aligned 2d bounding box
            return 1 - (min_height_fraction * min_width_fraction)

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(boxA: BoundingBoxRotated, boxB: BoundingBoxRotated):
        intersect_poly = intersection(boxA._polygon, boxB._polygon)
        if intersect_poly.is_empty:
            return False
        return True

    @staticmethod
    def get_intersection_area(boxA, boxB):
        intersect_poly = intersection(boxA._polygon, boxB._polygon)
        if intersect_poly.is_empty:
            return 0
        interArea = intersect_poly.area
        assert interArea >= 0
        return interArea

    @staticmethod
    def get_union_areas(
        boxA: BoundingBoxRotated, boxB: BoundingBoxRotated, interArea=None
    ):
        union_poly = union(boxA._polygon, boxB._polygon)
        union_area = union_poly.area
        assert union_area >= 0
        return union_area

    @staticmethod
    def get_amount_bounding_box_all_classes(
        bounding_boxes: list[BoundingBoxRotated], reverse=False
    ):
        classes = list(set([bb.get_class_id() for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(
                BoundingBoxRotated.get_bounding_box_by_class(bounding_boxes, c)
            )
        # Sort dictionary by the amount of bounding boxes
        ret = {
            k: v
            for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)
        }
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes: list[BoundingBoxRotated], class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(
        bounding_boxes: list[BoundingBoxRotated], image_name
    ):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes: list[BoundingBoxRotated]):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes: list[BoundingBoxRotated]):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)
