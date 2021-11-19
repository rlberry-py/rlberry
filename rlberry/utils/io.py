
import os
import zipfile
import pathlib


def zipdir(dir_path, ouput_fname):
    """
    Zip a directory.

    Parameters
    ----------
    dir_path : Path or str
        Directory to be compressed.
    output_fname : str
        Name of output zip file.

    Returns
    -------
    path to zip file, or None if dir_path does not exist.
    """
    dir_path = pathlib.Path(dir_path)
    if not dir_path.exists():
        return None
    ouput_fname = pathlib.Path(ouput_fname).with_suffix('.zip')
    zipf = zipfile.ZipFile(ouput_fname, 'w', zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(dir_path):
        for file in files:
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(dir_path, '..')))
    zipf.close()
    return ouput_fname
