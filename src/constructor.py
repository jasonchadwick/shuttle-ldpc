from src.GBCode import GBCode
from src.HGPCode import HGPCode
from src.HypercubeCode import HypercubeCode
from src.RadialCode import RadialCode
from src.RotatedSurfaceCode import RotatedSurfaceCode
from src.SHYPSCode import SHYPSCode
from src.TileCode import TileCode

def construct_code(name, *args):
    if name == 'BB':
        l,m,a_ord,b_ord = args
        code = GBCode(l,m,a_ord,b_ord)
    elif name == 'Simplex':
        r, = args
        code = HGPCode.simplex_code(r)
    elif name == 'Hypercube':
        r, = args
        code = HypercubeCode(r)
    elif name == 'Radial':
        r,s,d = args
        code = RadialCode(r,s,d)
    elif name == 'Surface':
        d, = args
        code = RotatedSurfaceCode(d)
    elif name == 'Tile':
        n,k,d = args
        code = TileCode.known_code(n,k,d)
    else:
        raise ValueError('Unknown code name', name)
    return code.to_qldpc_code_object()