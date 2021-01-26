"""三角形有关的功能"""

from typing import List
from .point import Point, middle_point
from .line import Segment

class Triangle():
    """三角形"""
    def __init__(self, pts: List[Point]):
        if len(pts) != 3:
            raise ValueError('三角形只能有三个点')
        self._pts = pts
        self._edges = [
            Segment(self._pts[0], self._pts[1]),
            Segment(self._pts[1], self._pts[2]),
            Segment(self._pts[2], self._pts[0]),
        ]

    def __str__(self):
        return '{%s:\n\t' % self.__class__.__name__+\
            str(self._pts[0]) + "\n\t" +\
            str(self._pts[1]) + "\n\t" +\
            str(self._pts[2]) + "}\n"

    def __repr__(self):
        return "{%s:\n\t" % self.__class__.__name__ +\
            repr(self._pts[0]) + "\n\t" +\
            repr(self._pts[1]) + "\n\t" +\
            repr(self._pts[2]) + "}\n"

    @property
    def vertex(self):
        '''角'''
        return self._pts

    @property
    def edges(self):
        '''边'''
        return self._edges


class Rtriangle(Triangle):
    """直角三角形\n
    第一个点需要是直角对应的顶点"""
    def __init__(self, rectver, ver1, ver2):
        super().__init__([rectver, ver1, ver2])
        self._center = middle_point(
            rectver,
            middle_point(ver1, ver2),
            sc1=1.,
            sc2=1.414)

    @property
    def center(self):
        '''中心点'''
        return self._center
