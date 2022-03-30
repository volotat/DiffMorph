import os
import numpy as np
from nilt_base.settingsreader import SetupTool
import morph_tools
from morph_tools import setup_morpher, single_image_morpher
from grating_helper import get_morphing_info_from_specs
from glob import glob

st = SetupTool("grating_morpher")
log = st.log


def fmod(n):
    if n < 5:
        n += 1
    return n


def bbox_midpoint(min_row, min_col, max_row, max_col):
    return (min_row + max_row) / 2, (min_col + max_col) / 2


specification_path = st.read_input("morph.specifications", datatype=str)
positions_path = st.read_input("morph.positions", datatype=str)
optimized_dir = st.read_input("morph.optimized_dir", datatype=str)
image_name = st.read_input("morph.image_name", datatype=str)
scale = st.read_input("morph.scale", datatype=float)
select_ids = st.read_input("morph.select_ids", datatype=list, default_value=False)

try:
    st.read_input("morph.parameters.im_sz")
    raise ValueError("morph.parameters.im_sz must not be set. It is inferred from specifications")
except ValueError:
    pass

parameters = st.settings.get("morph").get("parameters", None)

specs = get_morphing_info_from_specs(specification_path, positions_path, n_mod_fun=fmod)

if select_ids:
    selected = {}
    for sel_id in select_ids:
        selected[sel_id] = specs[sel_id]
    specs = selected

gen_outdir = st.output_folder
count = 0
for name_id, spec in specs.items():
    count += 1
    log.info(f"Processing {name_id} - {count}/{len(specs)}")
    # try:
    folders = glob(os.path.join(optimized_dir, name_id + "_*"))
    assert len(folders) == 2, f"Got more than two folder for single key: {folders}"
    target_image = os.path.join(optimized_dir, name_id + "_left", image_name)
    source_image = os.path.join(optimized_dir, name_id + "_right", image_name)
    assert os.path.isfile(source_image), f"Source image, {source_image}, does not exist"
    assert os.path.isfile(target_image), f"Source image, {target_image}, does not exist"

    # Split images into individual blobs
    src = morph_tools.load_image(source_image)
    trg = morph_tools.load_image(target_image)

    src_keypoints = morph_tools.find_blobs(src)
    trg_keypoints = morph_tools.find_blobs(trg)

    if not len(src_keypoints) == len(trg_keypoints):
        raise ValueError(f"Different number of Blobs found between source and target. "
                         f"{len(src_keypoints)} vs {len(trg_keypoints)}")

    src_midpoints = [bbox_midpoint(*blob.bbox) for blob in src_keypoints]
    trg_midpoints = [bbox_midpoint(*blob.bbox) for blob in trg_keypoints]

    src_blob_files = []
    src_folder = os.path.dirname(source_image)
    for i, blob in enumerate(src_keypoints):
        blob_file = os.path.join(src_folder, f"src_blob_{i:02d}.png")
        src_blob_files.append(blob_file)
        morph_tools.save_image(blob_file, blob.image, detect_range=True)

    trg_blob_files = []
    trg_folder = os.path.dirname(target_image)
    for i, blob in enumerate(trg_keypoints):
        blob_file = os.path.join(trg_folder, f"trg_blob_{i:02d}.png")
        trg_blob_files.append(blob_file)
        morph_tools.save_image(blob_file, blob.image, detect_range=True)

    # Create general folder
    id_folder = os.path.join(gen_outdir, f"morphed_{name_id}")
    if not os.path.isdir(id_folder):
        os.makedirs(id_folder)

    needed_dimensions = [(l, w) for l, w in zip(spec["lengths"], spec["widths"])]
    source_dim = (spec["lengths"][-1], spec["widths"][-1])
    target_dim = (spec["lengths"][0], spec["widths"][0])

    # CCreate empty images
    morphed_images = [np.zeros((int(np.ceil(l / scale)), int(np.ceil(w / scale)), 3)) for l, w in needed_dimensions]
    pct = morph_tools.interpolate_pct(spec["lengths"], source_dim[0], target_dim[0])

    # Go though each blob and morph it
    for i in range(len(src_keypoints)):
        src_blob = src_blob_files[i]
        trg_blob = trg_blob_files[i]
        blob_folder = os.path.join(id_folder, f"blob_{i:02d}")
        if not os.path.isdir(blob_folder):
            os.makedirs(blob_folder)

        morph_class_trained = setup_morpher(src_blob, trg_blob, output_folder=blob_folder, **parameters)

        log.info(f"Generating {len(needed_dimensions)}")
        for j in range(len(morphed_images)):

            # Get morphing for this dimension and insert into the placeholder image
            im = morph_tools.single_blob_morpher(morph_class_trained, pct[j], save_images=True)
            offset_h = int(np.round(src_midpoints[i][0] + (trg_midpoints[i][0] - src_midpoints[i][0]) * pct[j]))
            offset_w = int(np.round(src_midpoints[i][1] + (trg_midpoints[i][1] - src_midpoints[i][1]) * pct[j]))

            h, w = im.shape[0:2]
            vtop, vbot = morph_tools._get_vpos_idx(offset_h * 2, h, "c")
            hleft, hright = morph_tools._get_hpos_idx(offset_w * 2, w, "c")

            morphed_images[j][vtop:vbot, hleft:hright, ...] = im

    for i, im in enumerate(morphed_images):
        im_name = os.path.join(id_folder,
                               f'morphed_{i:03d}_{pct[i]:0.2f}pct_'
                               f'{needed_dimensions[i][0]:0.3f}x{needed_dimensions[i][1]:0.3f}.png')
        morph_tools.save_image(im_name, im)


        # except Exception as e:
        #    log.error(f"Processing of {name_id} FAILED! Trying next! Got error: {e}")

log.info(f"Done, output saved to: {gen_outdir}")
