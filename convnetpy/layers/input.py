from convnetpy.util import getopt
from convnetpy.vol import Vol

class InputLayer(object):

    def __init__(self, opt={}):
        self.out_sx = opt['out_sx']
        self.out_sy = opt['out_sy']
        self.out_depth = opt['out_depth']
        self.layer_type = 'input'

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
        n = len(V.w)
        for i in range(n):
            A.w[i] = V.w[i]
        self.out_act = A
        return self.out_act

    def backward(self):
        pass

    def getParamsAndGrads(self):
        return []

    def toJSON(self):
        return {
            'out_depth' : self.out_depth,
            'out_sx'    : self.out_sx,
            'out_sy'    : self.out_sy,
            'layer_type': self.layer_type
        }

    def fromJSON(self, json):
        self.out_depth  = json['out_depth']
        self.out_sx     = json['out_sx']
        self.out_sy     = json['out_sy']
        self.layer_type = json['layer_type']