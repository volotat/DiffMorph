import os
import numpy as np
import morph_tools
from nilt_base.settingsreader import SetupTool
from morph_tools import setup_morpher, single_image_morpher, single_image_morpher_resize
from grating_helper import get_morphing_info_from_specs
from glob import glob
import cv2
from run_manual_splitting import find_best_splitpoint

st = SetupTool("grating_morpher")
log = st.log


def fmod(n):
    if n < 5:
        n += 1
    return n


specification_path = st.read_input("morph.specifications", datatype=str)
positions_path = st.read_input("morph.positions", datatype=str)
optimized_dir = st.read_input("morph.optimized_dir", datatype=str)
image_name = st.read_input("morph.image_name", datatype=str)
scale = st.read_input("morph.scale", datatype=float)
split_n = st.read_input("morph.split_n", datatype=float, default_value=9)
try:
    st.read_input("morph.parameters.im_sz")
    raise ValueError("morph.parameters.im_sz must not be set. It is inferred from specifications")
except ValueError:
    pass

parameters = st.settings.get("morph").get("parameters", None)

specs = get_morphing_info_from_specs(specification_path, positions_path, n_mod_fun=fmod)

gen_outdir = st.output_folder
resized_folder = os.path.join(gen_outdir, "run_resized")
if not os.path.exists(resized_folder):
    os.mkdir(resized_folder)
i = 0
for name_id, spec in specs.items():
    i += 1
    log.info(f"Processing {name_id} - {i}/{len(specs)}")
    # try:
    folders = glob(os.path.join(optimized_dir, name_id + "_*"))
    assert len(folders) == 2, f"Got more than two folder for single key: {folders}"
    target_image = os.path.join(optimized_dir, name_id + "_left", image_name)
    source_image = os.path.join(optimized_dir, name_id + "_right", image_name)
    assert os.path.isfile(source_image), f"Source image, {source_image}, does not exist"
    assert os.path.isfile(target_image), f"Source image, {target_image}, does not exist"

    src_org = morph_tools.load_image(source_image)
    trg_org = morph_tools.load_image(target_image)
    src_resized = cv2.cvtColor(
        cv2.resize(cv2.cvtColor(src_org, cv2.COLOR_RGB2BGR), (src_org.shape[0], src_org.shape[0])), cv2.COLOR_BGR2RGB)
    trg_resized = cv2.cvtColor(
        cv2.resize(cv2.cvtColor(trg_org, cv2.COLOR_RGB2BGR), (trg_org.shape[0], trg_org.shape[0])), cv2.COLOR_BGR2RGB)
    src_name = os.path.join(resized_folder, name_id + "_right_resized.png")
    trg_name = os.path.join(resized_folder, name_id + "_left_resized.png")
    morph_tools.save_image(src_name, src_resized)
    morph_tools.save_image(trg_name, trg_resized)

    if spec["N"] >= split_n:
        src_org = morph_tools.load_image(src_name)
        trg_org = morph_tools.load_image(trg_name)
        pct = find_best_splitpoint(src_org, trg_org)
        log.info(f"DEBUG: Splitting image at {pct * 100:.2f} pct")
        src_dim1 = int(np.ceil(src_org.shape[0] * pct))
        trg_dim1 = int(np.ceil(trg_org.shape[0] * pct))
        src_splt1 = src_org[:src_dim1, ...]
        src_splt2 = src_org[src_dim1:, ...]
        trg_splt1 = trg_org[:trg_dim1, ...]
        trg_splt2 = trg_org[trg_dim1:, ...]

        src_name1 = os.path.join(resized_folder, name_id + "_right_split_top.png")
        src_name2 = os.path.join(resized_folder, name_id + "_right_split_bot.png")
        trg_name1 = os.path.join(resized_folder, name_id + "_left_split_top.png")
        trg_name2 = os.path.join(resized_folder, name_id + "_left_split_bot.png")

        morph_tools.save_image(src_name1, src_splt1)
        morph_tools.save_image(src_name2, src_splt2)
        morph_tools.save_image(trg_name1, trg_splt1)
        morph_tools.save_image(trg_name2, trg_splt2)

        id_folder = os.path.join(gen_outdir, f"morphed_{name_id}_top")
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)

        needed_dimensions = [(l, w) for l, w in zip(spec["lengths"] * pct, spec["widths"])]
        source_dim = (spec["lengths"][-1] * pct, spec["widths"][-1])
        target_dim = (spec["lengths"][0] * pct, spec["widths"][0])

        morph_class_trained = setup_morpher(src_name1, trg_name1, output_folder=id_folder, **parameters)

        log.info(f"Generating {len(needed_dimensions)} images for top part")
        for j, dim in enumerate(needed_dimensions):
            name = name_id + f"_im_{j:03d}"
            im = single_image_morpher_resize(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
                                             save_images=True)

        id_folder = os.path.join(gen_outdir, f"morphed_{name_id}_bot")
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)

        needed_dimensions = [(l, w) for l, w in zip(spec["lengths"] * (1 - pct), spec["widths"])]
        source_dim = (spec["lengths"][-1] * (1 - pct), spec["widths"][-1])
        target_dim = (spec["lengths"][0] * (1 - pct), spec["widths"][0])

        morph_class_trained = setup_morpher(src_name2, trg_name2, output_folder=id_folder, **parameters)

        log.info(f"Generating {len(needed_dimensions)} images for bottom part")
        for j, dim in enumerate(needed_dimensions):
            name = name_id + f"_im_{j:03d}"
            im = single_image_morpher_resize(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
                                             save_images=True)
    else:

        id_folder = os.path.join(gen_outdir, f"morphed_{name_id}")
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)

        needed_dimensions = [(l, w) for l, w in zip(spec["lengths"], spec["widths"])]
        source_dim = (spec["lengths"][-1], spec["widths"][-1])
        target_dim = (spec["lengths"][0], spec["widths"][0])

        morph_class_trained = setup_morpher(src_name, trg_name, output_folder=id_folder, **parameters)

        log.info(f"Generating {len(needed_dimensions)} images")
        for j, dim in enumerate(needed_dimensions):
            name = name_id + f"_im_{j:03d}"
            im = single_image_morpher_resize(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
                                             save_images=True)
        # except Exception as e:
        #    log.error(f"Processing of {name_id} FAILED! Trying next! Got error: {e}")

log.info(f"Done, output saved to: {gen_outdir}")
