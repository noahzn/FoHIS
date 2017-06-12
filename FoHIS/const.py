"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""

import sys


class Const(object):
    class ConstError(Exception):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError
        else:
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.key
        else:
            return None


sys.modules[__name__] = Const()
