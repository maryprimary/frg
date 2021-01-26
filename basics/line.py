"""和线有关的内容"""


#class Line():
#    """直线，没有端点"""
#    def __init__(self, *args):
#        raise NotImplementedError('没有实现')

from typing import Tuple
from .point import Point

class Segment():
    """线段，由两个端点来决定"""
    def __init__(self, pt1: Point, pt2: Point):
        self._pt1 = pt1
        self._pt2 = pt2
        self._iteridx = 0

    def __str__(self):
        return "{Segment:\n\t" +\
            str(self._pt1) + "\n\t" + str(self._pt2) + "}\n"

    def __repr__(self):
        return "(%s-%s)" % (repr(self._pt1), repr(self._pt2))

    @property
    def ends(self) -> Tuple[Point, Point]:
        '''线段的两个端点'''
        return [self._pt1, self._pt2]

    #def __getitem__(self, idx):
    #    if idx == 1:
    #        return self._pt1
    #    if idx == 2:
    #        return self._pt2
    #    raise ValueError('索引应该是1或者2')
#
    #def __iter__(self):
    #    self._iteridx = 0
    #    return self
#
    #def __next__(self):
    #    self._iteridx += 1
    #    if self._iteridx == 1:
    #        return self._pt1
    #    if self._iteridx == 2:
    #        return self._pt2
    #    raise StopIteration
