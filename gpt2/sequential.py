from module import ModuleList
from functools import reduce

class Sequential(ModuleList):
    def __init__(self, *layers):
        list.__init__(self, layers)

    def __call__(self, x):
        return reduce(lambda x, f: f(x), self, x)