# Flag to debug without screen 
_DEBUG_NO_SCREEN = False

def _activate_debug_mode():
    global _DEBUG_NO_SCREEN
    _DEBUG_NO_SCREEN = True

def _deactivate_debug_mode():
    global _DEBUG_NO_SCREEN
    _DEBUG_NO_SCREEN = False