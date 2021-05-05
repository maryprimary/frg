"""正六边形"""


import numpy
from .point import Point
from .line import Segment


class Hexagon():
    """正六边形，只能是正着摆的\n
    需要中点和中点到边的距离
    """
    def __init__(self, center: Point, height):
        if center.typeid != 1:
            raise ValueError('只能在直角坐标系')
        self._center = center
        xlim = 2 * height / numpy.sqrt(3)
        ylim = height
        self._height = height
        cnx, cny = center.coord[0], center.coord[1]
        self._vertex = [
            Point(cnx+xlim, cny, 1),
            Point(cnx+0.5*xlim, cny+ylim, 1),
            Point(cnx-0.5*xlim, cny+ylim, 1),
            Point(cnx-xlim, cny, 1),
            Point(cnx-0.5*xlim, cny-ylim, 1),
            Point(cnx+0.5*xlim, cny-ylim, 1)
        ]
        self._edges = [
            Segment(self._vertex[idx-1], self._vertex[idx])\
                for idx in range(1, 6)
        ]
        self._edges.append(
            Segment(self._vertex[5], self._vertex[0])
        )
        #求出面积
        self._width = 6 * self._height / numpy.sqrt(3)

    def __str__(self):
        return "{%s:\n\t" % self.__class__.__name__ +\
            str(self._vertex[0]) + "\n\t" +\
            str(self._vertex[1]) + "\n\t" +\
            str(self._vertex[2]) + "\n\t" +\
            str(self._vertex[3]) + "\n\t" +\
            str(self._vertex[4]) + "\n\t" +\
            str(self._vertex[5]) + "}\n"

    def __repr__(self):
        return "{%s:\n\t" % self.__class__.__name__ +\
            repr(self._pts[0]) + "\n\t" +\
            repr(self._pts[1]) + "\n\t" +\
            repr(self._pts[2]) + "\n\t" +\
            repr(self._pts[3]) + "\n\t" +\
            repr(self._pts[4]) + "\n\t" +\
            repr(self._pts[5]) + "}\n"


    @property
    def vertex(self):
        '''顶点'''
        return self._vertex

    @property
    def edges(self):
        '''边'''
        return self._edges

    @property
    def center(self):
        '''中心'''
        return self._center

    @property
    def height(self):
        '''创建六边形时的高度'''
        return self._height

    @property
    def width(self):
        '''这个宽度没有实际的意义，只是为了保证
        面积等于高乘以宽
        '''
        return self._width
