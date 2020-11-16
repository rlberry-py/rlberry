# Flag to debug without screen 
_DEBUG_NO_SCREEN = False

def _activate_debug_mode():
    global _DEBUG_NO_SCREEN
    _DEBUG_NO_SCREEN = True


def _deactivate_debug_mode():
    global _DEBUG_NO_SCREEN
    _DEBUG_NO_SCREEN = False


from .core import Scene, GeometricPrimitive
from .render_interface import RenderInterface
from .render_interface import RenderInterface2D

