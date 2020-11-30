
#
#  Code to visualize the environments.
#

import base64
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
# from IPython.display import clear_output
from pathlib import Path


def show_video(filename=None, directory='./videos'):
    """
    Either show all videos in a directory (if filename is None) or
    show video corresponding to filename.
    """
    html = []
    if filename is not None:
        files = Path('./').glob(filename)
    else:
        files = Path(directory).glob("*.mp4")
    for mp4 in files:
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


display = Display(visible=0, size=(1400, 900))
display.start()
