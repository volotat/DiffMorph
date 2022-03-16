import numpy as np
from morphing import Morph
import cv2
from nilt_base.NILTlogger import get_logger

logger = get_logger(__name__)

COLOR_LOOKUP = {"black": [0, 0, 0],
                "white": [255, 255, 255],
                "red": [255, 0, 0],
                "green": [0, 255, 0],
                "blue": [0, 0, 255]}


def load_image(file):
    """ Load an image into a numpy array from file

    Parameters
    ----------
    file : str
        Path to image file

    Returns
    -------
    np.ndarray

    """
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save_image(path, array):
    """ Save a numpy array as a RGB image.

    Parameters
    ----------
    path : str
        Path where to save file
    array : np.ndarray
        image (as a numpy array) to save.

    Returns
    -------

    """
    succes = cv2.imwrite(path, cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not succes:
        raise RuntimeError("Image not saved sucessfully")


def morph(source, target, steps, output_folder, **kwargs):
    """

    Parameters
    ----------
    source : str
        Path to the source image
    target : str
        Path to the target image
    steps :
        Number of images wanted in the sequence
    output_folder : str
        Folder to save results to
    kwargs
        Keyword arguments passed to Morph class

    Returns
    -------
    list
        List of paths to PNG images generated
    list
        List of paths to Numpy arrays of images generated
    """

    mc = Morph(output_folder=output_folder, **kwargs)
    source = mc.load_image(source)
    target = mc.load_image(target)

    mc.produce_warp_maps(source, target)
    png_image_paths, npy_image_paths = mc.use_warp_maps(source, target, steps)

    return png_image_paths, npy_image_paths


def crop_image_to_size(image, size, scale, pos="cc"):
    """

    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    size : tuple
        Tuple (height, width) of are to get height and width in um
    scale : float
        Resolutions of image as: um pr pixel
    pos : str
        Relative vertical position image, first character gives vertical position, second gives horisontal position
        Vertical position should be [t]op, [c]enter or [b]ottom.
        Horisontal position should be [l]eft, [c]enter or [r]ight

    Returns
    -------
    np.ndarray
        Cropped image
    """
    assert isinstance(size, (tuple, int, np.ndarray)), f"Argument 'size' be a tuple, not {type(size).__name__}"
    assert len(size) == 2, f"Argument 'size' must have length 2, not {len(size)}"
    vpx = int(np.ceil(size[0] / scale))
    hpx = int(np.ceil(size[1] / scale))

    (vsize, hsize, csize) = image.shape
    if (vpx, hpx) > (vsize, hsize):
        logger.warning(f"At least one dimension of {(vpx, hpx)} is larger than {image.shape}, this dimension"
                       f"will limited to size of image.")
        vpx = min(vpx, image.shape[0])
        hpx = min(hpx, image.shape[1])

    assert isinstance(pos, str), f"Argument 'pos' be a string, not {type(pos).__name__}"
    assert len(pos) == 2, f"Argument 'pos' must have length 2, not {len(pos)}"
    vpos = pos[0].lower()
    hpos = pos[1].lower()
    assert vpos in "tcb", f"First element of 'pos' must be either, 't', 'c' or 'b' not {vpos}"
    assert hpos in "lcr", f"First element of 'pos' must be either, 'l', 'c' or 'r' not {vpos}"

    vtop, vbot = _get_vpos_idx(vsize, vpx, vpos)
    hleft, hright = _get_vpos_idx(hsize, hpx, hpos)

    crop = image[vtop:vbot, hleft:hright, ...]
    # Opposite of crop
    # inverse = np.copy(image)
    # inverse[vtop:vbot, hleft:hright, ...] = [0, 0, 255]
    return crop


def pad_image_to_square(image, size=None, color="black", pos="cc"):
    """ Pad an image to make it a square

    Parameters
    ----------
    image : np.ndarray
        Image to pad
    size : int (optional)
        Size to pad to.
    color : string or list
        Color to pad with either a string "black", "red". Default: "black"
    pos : str
        Relative vertical position image, first character gives vertical position, second gives horisontal position
        Vertical position should be [t]op, [c]enter or [b]ottom.
        Horisontal position should be [l]eft, [c]enter or [r]ight

    Returns
    -------

    """

    assert isinstance(color, str), f"Expected 'color' to be of type 'str' not '{type(color).__name__}'."
    assert color.lower() in COLOR_LOOKUP, f"Color '{color}' not found."
    bg = COLOR_LOOKUP.get(color)

    assert isinstance(pos, str), f"Expected argument 'pos' to be of type 'str', not {type(pos).__name__}"
    assert len(pos) == 2, f"Argument 'pos' must have length 2, not {len(pos)}"
    vpos = pos[0].lower()
    hpos = pos[1].lower()
    assert vpos in "tcb", f"First element of 'pos' must be either, 't', 'c' or 'b' not {vpos}"
    assert hpos in "lcr", f"First element of 'pos' must be either, 'l', 'c' or 'r' not {vpos}"

    assert image.ndim == 3, "image must be RGB"
    assert image.shape[2] == 3, "image must be RGB"

    if size is None:
        size = max(image.shape[0:2])
    (vsize, hsize, csize) = image.shape
    if not (size, size) > (vsize, hsize):
        msg = f"At least one dimension of the image {image.shape} is larger than the specified size of {size}."
        logger.error(msg)
        raise ValueError(msg)

    padded = np.full((size, size, 3), fill_value=bg)

    vtop, vbot = _get_vpos_idx(size, vsize, vpos)
    hleft, hright = _get_vpos_idx(size, hsize, hpos)

    # Debug
    if not padded[vtop:vbot, hleft:hright, :].shape == image.shape:
        raise RuntimeError("Something went wrong cutout does not match image")

    padded[vtop:vbot, hleft:hright, :] = image

    return padded


def pad_images_to_same_square(*images, **kwargs):
    size = kwargs.get("size", None)
    if size is None:
        size = 0
        for im in images:
            size = max(size, max(im.shape[:2]))
        kwargs["size"] = size

    all_padded = []
    for im in images:
        all_padded.append(pad_image_to_square(im, **kwargs))
    return all_padded

def _get_vpos_idx(large_size, small_size, pos):
    if pos == "c":
        vcenter = large_size // 2
        vtop = int(np.floor(vcenter - small_size / 2))
        vbot = int(np.floor(vcenter + small_size / 2))
    elif pos == "b":
        vtop = large_size - small_size
        vbot = large_size
    elif pos == "t":
        vtop = 0
        vbot = small_size
    else:
        raise RuntimeError(f"Unexpected vpos of {pos}")

    return vtop, vbot


def _get_hpos_idx(large_size, small_size, pos):
    if pos == "c":
        hcenter = large_size // 2
        hleft = int(np.floor(hcenter - small_size / 2))
        hright = int(np.floor(hcenter + small_size / 2))
    elif small_size == "l":
        hleft = 0
        hright = small_size
    elif pos == "r":
        hleft = large_size - small_size
        hright = large_size
    else:
        raise RuntimeError(f"Unexpected hpos of {pos}")

    return hleft, hright
