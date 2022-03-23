import os
import numpy as np
from nilt_base.settingsreader import SetupTool

import morph_tools
from morph_tools import setup_morpher, single_image_morpher
from grating_helper import get_morphing_info_from_specs
from glob import glob

import matplotlib.pyplot as plt

st = SetupTool("grating_morpher")
log = st.log.logger


def fmod(n):
    if n < 5:
        n += 1
    return n


def find_best_splitpoint(im1, im2):
    if im1.ndim == 3:
        im1 = np.any(im1, axis=2)
    if im2.ndim == 3:
        im2 = np.any(im2, axis=2)
    line1 = np.invert(np.any(im1, axis=1))
    line2 = np.invert(np.any(im2, axis=1))
    rel1 = np.linspace(0, 1, len(line1))
    rel2 = np.linspace(0, 1, len(line2))
    components1 = []
    i = 0
    while i < len(line1):
        comp = []
        while (i < len(line1)) and line1[i]:
            comp.append(rel1[i])
            i += 1
        if comp:
            components1.append(comp)
        i += 1

    components2 = []

    i = 0
    while i < len(line2):
        comp = []
        while (i < len(line2)) and line2[i]:
            comp.append(rel2[i])
            i += 1
        if comp:
            components2.append(comp)
        i += 1

    means1 = np.asarray([np.mean(arr) for arr in components1])
    means2 = np.asarray([np.mean(arr) for arr in components2])

    components1 = [x for _, x in sorted(zip(np.abs(means1 - 0.5), components1))]
    components2 = [x for _, x in sorted(zip(np.abs(means2 - 0.5), components2))]

    common_mean = None
    for comp1 in components1:
        mean1 = np.mean(comp1)
        max1 = np.max(comp1)
        min1 = np.min(comp1)

        for comp2 in components2:
            mean2 = np.mean(comp2)
            max2 = np.max(comp2)
            min2 = np.min(comp2)
            if (min1 < mean2 < max1) and (min2 < mean1 < max2):
                common_mean = np.mean([mean1, mean2])
            elif min1 < mean2 < max1:
                common_mean = mean2
            elif min2 < mean1 < max2:
                common_mean = mean2
            elif min2 < min1 < max2:
                common_mean = np.mean([min1, max2])
            elif min2 < max1 < max2:
                common_mean = np.mean([max1, min2])
            if common_mean:
                break
        if common_mean:
            break

    #plt.plot(rel1, np.invert(line1), "*")
    #plt.plot(rel2, np.invert(line2), "*")
    #plt.axline((common_mean, 1), (common_mean,0), color="red")
    #plt.show()

    if common_mean is None:
        raise ValueError(f"Did not find overlapping mean")

    return common_mean


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
i = 0
for name_id, spec in specs.items():
    i += 1
    log.info(f"Processing {name_id} - {i}/{len(specs)}")
    # try:
    folders = glob(os.path.join(optimized_dir, name_id + "_*"))
    assert len(folders) > 0, f"Did not find any folders for key: {name_id}"
    assert len(folders) == 2, f"Got more than two folder for single key: {folders}"
    target_image = os.path.join(optimized_dir, name_id + "_left", image_name)
    source_image = os.path.join(optimized_dir, name_id + "_right", image_name)
    assert os.path.isfile(source_image), f"Source image, {source_image}, does not exist"
    assert os.path.isfile(target_image), f"Source image, {target_image}, does not exist"

    FAILED = False
    if spec["N"] >= split_n:
        src_org = morph_tools.load_image(source_image)
        trg_org = morph_tools.load_image(target_image)
        pct = find_best_splitpoint(src_org, trg_org)
        log.info(f"DEBUG: Splitting image at {pct*100:.2f} pct")
        src_dim1 = int(np.ceil(src_org.shape[0] * pct))
        trg_dim1 = int(np.ceil(trg_org.shape[0] * pct))
        src_splt1 = src_org[:src_dim1, ...]
        src_splt2 = src_org[src_dim1:, ...]
        trg_splt1 = trg_org[:trg_dim1, ...]
        trg_splt2 = trg_org[trg_dim1:, ...]

        src_name1 = os.path.join(optimized_dir, name_id + "_right", "split_top.png")
        src_name2 = os.path.join(optimized_dir, name_id + "_right", "split_bot.png")
        trg_name1 = os.path.join(optimized_dir, name_id + "_left", "split_top.png")
        trg_name2 = os.path.join(optimized_dir, name_id + "_left", "split_bot.png")

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
            im = single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
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
            im = single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
                                      save_images=True)
    else:

        id_folder = os.path.join(gen_outdir, f"morphed_{name_id}")
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)

        needed_dimensions = [(l, w) for l, w in zip(spec["lengths"], spec["widths"])]
        source_dim = (spec["lengths"][-1], spec["widths"][-1])
        target_dim = (spec["lengths"][0], spec["widths"][0])

        morph_class_trained = setup_morpher(source_image, target_image, output_folder=id_folder, **parameters)

        log.info(f"Generating {len(needed_dimensions)} images")
        for j, dim in enumerate(needed_dimensions):
            name = name_id + f"_im_{j:03d}"
            im = single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, name=name,
                                      save_images=True)
        # except Exception as e:
        #    log.error(f"Processing of {name_id} FAILED! Trying next! Got error: {e}")

log.info(f"Done, output saved to: {gen_outdir}")
