"""长方形有关的内容"""

#from typing import List
from .point import Point, shift_point
from .line import Segment

class Rectangle():
    """一个一般的长方形"""
    def __init__(self, center: Point, width, height):
        self._width = width
        self._height = height
        self._center = center
        self._pts = [
            shift_point(self._center, -width/2, height/2),
            shift_point(self._center, -width/2, -height/2),
            shift_point(self._center, width/2, -height/2),
            shift_point(self._center, width/2, height/2)
        ]
        self._edges = [
            Segment(self._pts[0], self._pts[1]),
            Segment(self._pts[1], self._pts[2]),
            Segment(self._pts[2], self._pts[3]),
            Segment(self._pts[3], self._pts[0]),
        ]

    def __str__(self):
        return "{%s:\n\t" % self.__class__.__name__ +\
            str(self._pts[0]) + "\n\t" +\
            str(self._pts[1]) + "\n\t" +\
            str(self._pts[2]) + "\n\t" +\
            str(self._pts[3]) + "}\n"


    def __repr__(self):
        return "{%s:\n\t" % self.__class__.__name__ +\
            repr(self._pts[0]) + "\n\t" +\
            repr(self._pts[1]) + "\n\t" +\
            repr(self._pts[2]) + "\n\t" +\
            repr(self._pts[3]) + "}\n"

    @property
    def vertex(self):
        '''角'''
        return self._pts

    @property
    def edges(self):
        '''边'''
        return self._edges

    @property
    def center(self):
        '''中心点'''
        return self._center

    @property
    def width(self):
        '''宽度'''
        return self._width

    @property
    def height(self):
        '''高度'''
        return self._height


class Square(Rectangle):
    """正方形"""
    def __init__(self, center: Point, width):
        super().__init__(center, width, width)
