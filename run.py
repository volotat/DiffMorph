import os
import numpy as np
from nilt_base.settingsreader import SetupTool
from morph_tools import setup_morpher, single_image_morpher
from grating_helper import get_morphing_info_from_specs
from glob import glob

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
select_ids = st.read_input("morph.select_ids", datatype=list, default_value=None)

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
i = 0
for name_id, spec in specs.items():
    i += 1
    log.info(f"Processing {name_id} - {i}/{len(specs)}")
    #try:
    folders = glob(os.path.join(optimized_dir, name_id + "_*"))
    assert len(folders) == 2, f"Got more than two folder for single key: {folders}"
    target_image = os.path.join(optimized_dir, name_id + "_left", image_name)
    source_image = os.path.join(optimized_dir, name_id + "_right", image_name)
    assert os.path.isfile(source_image), f"Source image, {source_image}, does not exist"
    assert os.path.isfile(target_image), f"Source image, {target_image}, does not exist"

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
        im = single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, name=name, save_images=True)
    #except Exception as e:
    #    log.error(f"Processing of {name_id} FAILED! Trying next! Got error: {e}")

log.info(f"Done, output saved to: {gen_outdir}")
