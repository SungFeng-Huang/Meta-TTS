from .fit import XvecFitSystem
from .validate import XvecValidateSystem
from .test import XvecTestSystem

class XvecSystem(XvecFitSystem, XvecValidateSystem, XvecTestSystem):
    pass
