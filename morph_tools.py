import logging
import os.path

import numpy as np
from morphing import Morph
import cv2
from nilt_base.NILTlogger import get_logger
from scipy.ndimage import gaussian_filter

from skimage import measure

logger = get_logger("NILTlogger.morph_tool")

COLOR_LOOKUP = {"black": [0, 0, 0],
                "white": [255, 255, 255],
                "red": [255, 0, 0],
                "green": [0, 255, 0],
                "blue": [0, 0, 255]}


def make_image_rgb(image):
    """ Takes a grayscale numpy image and makes it RGB

    If image already has a 3'rd dim with size 3, it does nothing.

    Parameters
    ----------
    image : np.ndarray
        Image to make RGB

    Returns
    -------
    np.ndarray
        array with shape (h,w,3)

    """
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f"Number of dimensions should be 2 or 3, not {image.ndim}")

    if image.ndim == 2:
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)

    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    elif image.shape[2] == 3:
        return image

    raise ValueError(f"Could not convert image with shape {image.shape}")


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
    if not os.path.exists(file):
        raise FileNotFoundError(f"No such file: {file}")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"{file} is not a file")
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save_image(path, array, detect_range=True):
    """ Save a numpy array as a RGB image.

    Parameters
    ----------
    path : str
        Path where to save file
    array : np.ndarray
        image (as a numpy array) to save.
    detect_range : bool (optional)
        If it should be detected whether the scale is [0,1] or [0, 255], and tru to convert to [0, 255]

    Returns
    -------

    """
    if not os.path.isdir(os.path.dirname(path)):
        raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist.")

    if detect_range:
        if np.max(array) <= 1.0:
            logger.debug(f"Fixing range before saving {path}")
            array = array * 255

    array = make_image_rgb(array)
    succes = cv2.imwrite(path, cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not succes:
        raise RuntimeError("Image not saved sucessfully")
    else:
        logger.debug(f"Successfully saved image to {path}")


def interpolate_pct(wanted, source, target):
    if source > target:
        raise ValueError("Source should be smaller than target value")
    return (wanted - source) / (target - source)


def bbox_midpoint(min_row, min_col, max_row, max_col):
    """ Get the midpoint of a bounding box

    Parameters
    ----------
    min_row : int
    min_col : int
    max_row : int
    max_col : int

    Returns
    -------
    tuple
        Midpoint af the bounding box, tuple of (middle row, middle col) as floats

    """
    return (min_row + max_row) / 2, (min_col + max_col) / 2


def find_blobs(image):
    """ Find individual structures in an im
    
    Parameters
    ----------
    image : cv2 image

    Returns
    -------
    skimage.measure.regionprops
    """
    if np.ndim(image) == 3:
        image = np.mean(image, axis=2)
    binary = image > 127.5
    blob_labels = measure.label(binary)
    blob_features = measure.regionprops(blob_labels)

    return blob_features


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
    source = mc.load_image_file(source)
    target = mc.load_image_file(target)

    mc.produce_warp_maps(source, target)
    png_image_paths, npy_image_paths = mc.use_warp_maps(source, target, steps)

    return png_image_paths, npy_image_paths


def setup_morpher(source, target, output_folder, padding_args=None, **kwargs):
    """

    Parameters
    ----------
    source : str or np.ndarray
        Path to the source image or image as numpy array
    target : str or np.ndarray
        Path to the target image or image as numpy array
    steps :
        Number of images wanted in the sequence
    output_folder : str
        Folder to save results to
    kwargs
        Keyword arguments passed to Morph class

    Returns
    -------
    Morph
        Trained class for morphing

    """
    # Prepare images
    if isinstance(source, str):
        source = load_image(source)
    else:
        source = make_image_rgb(source)

    if isinstance(target, str):
        target = load_image(target)
    else:
        target = make_image_rgb(target)

    # Pad them to the same square size
    source, target = pad_images_to_same_square(source, target, color="black", **padding_args)

    src_name_padded = os.path.join(output_folder, "source_image_padded.png")
    trg_name_padded = os.path.join(output_folder, "target_image_padded.png")
    save_image(src_name_padded, source)
    save_image(trg_name_padded, target)

    im_size = source.shape[0]
    mc = Morph(output_folder=output_folder, im_sz=im_size, **kwargs)

    logger.info("Training model, this might take a while")
    source = mc.load_image_file(src_name_padded)
    target = mc.load_image_file(trg_name_padded)
    mc.produce_warp_maps(source, target)
    logger.info("Training Done")

    return mc


def single_image_morpher(morph_class, morphed_dim, source_dim, target_dim, scale, save_images=True, name=""):
    """

    Parameters
    ----------
    save_images
    morph_class : morphing.Morph
        A trained instance of the Morph class.
    morphed_dim : tuple
        Tuple (height, width) in um, dimensions of the wanted morphed image
    source_dim : tuple
        Tuple (height, width) in um, dimensions of the original source image
    target_dim : tuple
        Tuple (height, width) in um, dimensions of the original target
    scale : float
        Resolutions of image as: um pr pixel
    save_images : bool or string
        Folder to save images to default folder from morph_class or a specific folder
    name : str
        Name to be used for the file along with dimensions

    Returns
    -------
    np.ndarray
        Morhped image
    """

    for t in (morphed_dim, source_dim, target_dim):
        assert isinstance(t, (tuple, list, np.ndarray)), f"Dimensions must be given as 2-tuples, not {t}"
        assert len(t) == 2, f"Dimensions must have length 2, got a dimension of {t}"

    height_pct = interpolate_pct(morphed_dim[0], source_dim[0], target_dim[0])
    # width_pct = interpolate_pct(morphed_dim[1], source_dim[1], target_dim[1])

    # if not np.isclose(width_pct, height_pct):
    #    logger.debug("Relative height and width placement is not close. Using relative height.")
    height_pct = height_pct * 100
    morphed_im = morph_class.generate_single_morphed(height_pct)

    crop_im = crop_image_to_size(morphed_im, morphed_dim, scale)

    if save_images:
        if isinstance(save_images, str):
            outdir = save_images
        else:
            outdir = os.path.join(morph_class.output_folder, "single_morphed")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if not name:
            name = "single_morph"
        name += f"_{height_pct:.1f}pct_{morphed_dim[0]:.3f}x{morphed_dim[1]:.3f}.png"

        save_image(os.path.join(outdir, name), crop_im, detect_range=False)

    return crop_im


def single_blob_morpher(morph_class, pct, crop_threshold=10, save_images=True, name=""):
    """

    Parameters
    ----------
    morph_class : morphing.Morph
        A trained instance of the Morph class.
    pct : float
        Percentage of the transition between source and target, should be in the range [0, 1]
    crop_threshold : int or float
        Value between 0 and 255, threshold value for binarize image, when cropping out structure.
    save_images : bool or string
        Folder to save images to default folder from morph_class or a specific folder
    name : str
        Name to be used for the file along with dimensions

    Returns
    -------
    np.ndarray
        Morhped image
    """
    if pct > 1.0:
        ValueError(f"Morph percentage should be a float between [0, 1]. Got {pct}")

    height_pct = pct * 100
    morphed_im = morph_class.generate_single_morphed(height_pct)

    mean_im = np.mean(gaussian_filter(morphed_im, sigma=2), axis=2) >= crop_threshold
    rows = np.argwhere(np.sum(mean_im, axis=1))
    cols = np.argwhere(np.sum(mean_im, axis=0))
    min_row, max_row = np.min(rows), np.max(rows) + 1
    min_col, max_col = np.min(cols), np.max(cols) + 1

    crop_im = morphed_im[min_row:max_row, min_col:max_col, ...]

    if save_images:
        if isinstance(save_images, str):
            outdir = save_images
        else:
            outdir = os.path.join(morph_class.output_folder, "morhped_blob")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if not name:
            name = "single_blob"
        name += f"_{height_pct:.1f}pct.png"

        save_image(os.path.join(outdir, name), crop_im, detect_range=False)

    return crop_im


def single_image_morpher_resize(morph_class, morphed_dim, source_dim, target_dim, scale, save_images=True, name=""):
    """

    Parameters
    ----------
    save_images
    morph_class : morphing.Morph
        A trained instance of the Morph class.
    morphed_dim : tuple
        Tuple (height, width) in um, dimensions of the wanted morphed image
    source_dim : tuple
        Tuple (height, width) in um, dimensions of the original source image
    target_dim : tuple
        Tuple (height, width) in um, dimensions of the original target
    scale : float
        Resolutions of image as: um pr pixel
    save_images : bool or string
        Folder to save images to default folder from morph_class or a specific folder
    name : str
        Name to be used for the file along with dimensions

    Returns
    -------
    np.ndarray
        Morhped image
    """

    for t in (morphed_dim, source_dim, target_dim):
        assert isinstance(t, (tuple, list, np.ndarray)), f"Dimensions must be given as 2-tuples, not {t}"
        assert len(t) == 2, f"Dimensions must have length 2, got a dimension of {t}"

    height_pct = interpolate_pct(morphed_dim[0], source_dim[0], target_dim[0])
    # width_pct = interpolate_pct(morphed_dim[1], source_dim[1], target_dim[1])

    # if not np.isclose(width_pct, height_pct):
    #    logger.debug("Relative height and width placement is not close. Using relative height.")
    height_pct = height_pct * 100
    morphed_im = morph_class.generate_single_morphed(height_pct)

    crop_im = crop_image_to_size(morphed_im, (morphed_dim[0], morphed_dim[0]), scale)
    vpx = int(np.ceil(morphed_dim[0] / scale))
    hpx = int(np.ceil(morphed_dim[1] / scale))
    re_im = cv2.cvtColor(cv2.resize(cv2.cvtColor(crop_im, cv2.COLOR_RGB2BGR), (hpx, vpx)), cv2.COLOR_BGR2RGB)

    if save_images:
        if isinstance(save_images, str):
            outdir = save_images
        else:
            outdir = os.path.join(morph_class.output_folder, "single_morphed")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if not name:
            name = "single_morph"
        name += f"_{height_pct:.1f}pct_{morphed_dim[0]}x{morphed_dim[1]}.png"

        save_image(os.path.join(outdir, name), re_im, detect_range=False)

    return re_im


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


def pad_image_to_square(image, size=None, extra_pad=0, color="black", pos="cc"):
    """ Pad an image to make it a square

    Parameters
    ----------
    image : np.ndarray
        Image to pad
    size : int (optional)
        Size to pad to.
    extra_pad : int (optional)
        Number of extra pixels to each side. Default 0.
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

    assert isinstance(extra_pad, int), f"extra_pad must be an int, not {type(extra_pad).__name__}"
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
    size += 2 * extra_pad
    (vsize, hsize, csize) = image.shape
    if (size, size) < (vsize, hsize):
        msg = f"At least one dimension of the image {image.shape} is larger than the specified size of {size}."
        logger.error(msg)
        raise ValueError(msg)

    padded = np.full((size, size, 3), fill_value=bg)

    vtop, vbot = _get_vpos_idx(size, vsize, vpos)
    hleft, hright = _get_vpos_idx(size, hsize, hpos)

    # Debug
    if not padded[vtop:vbot, hleft:hright, :].shape == image.shape:
        raise RuntimeError(f"Something went wrong cutout does not match image. "
                           f"{padded[vtop:vbot, hleft:hright, :].shape} != {image.shape}")

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

    # limit to image size, no wrapping,
    if vtop < 0:
        vtop += 1
        vbot += 1

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

    # limit to image size, no wrapping,
    if hleft < 0:
        hleft += 1
        hright += 1

    return hleft, hright
