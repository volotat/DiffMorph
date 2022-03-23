import os
import numpy as np
import morph_tools
from nilt_base.settingsreader import SetupTool
from morph_tools import setup_morpher, single_image_morpher, single_image_morpher_resize
from grating_helper import get_morphing_info_from_specs
from glob import glob
import cv2
from itertools import product

st = SetupTool("Parameter_investigation")
log = st.log

source_im = st.read_input("morph.source_im", datatype=str)
target_im = st.read_input("morph.target_im", datatype=str)
specification_path = st.read_input("morph.specifications", datatype=str)
positions_path = st.read_input("morph.positions", datatype=str)
scale = st.read_input("morph.scale", datatype=float)

def fmod(n):
    if n < 5:
        n += 1
    return n


specs = get_morphing_info_from_specs(specification_path, positions_path, n_mod_fun=fmod)
spec = specs["N12_1"]

train_epochs = st.read_input("parameters.train_epochs", datatype=list, default_value=[1000])
mp_sz = st.read_input("parameters.mp_sz", datatype=list, default_value=[96])
warp_scale = st.read_input("parameters.warp_scale", datatype=list, default_value=[0.05])
mult_scale = st.read_input("parameters.mult_scale", datatype=list, default_value=[0.4])
add_scale = st.read_input("parameters.add_scale", datatype=list, default_value=[0.4])
add_first = st.read_input("parameters.add_first", datatype=list, default_value=[False])


names = ("train_epochs", "mp_sz", "warp_scale", "mult_scale", "add_scale", "add_first")

total = 1
for el in (train_epochs, mp_sz, warp_scale, mult_scale, add_scale, add_first):
    total *= len(el)
params = product(train_epochs, mp_sz, warp_scale, mult_scale, add_scale, add_first)



gen_outdir = st.output_folder
i = 0
for ps in params:
    i += 1
    log.info(f"Processing {i}/{total}")
    train_epochs, mp_sz, warp_scale, mult_scale, add_scale, add_first = ps

    naming = "params"
    kwargs = {}
    for n, v in zip(names, ps):
        naming += f"_{n}={v}"
        kwargs[n] = v

    folder = os.path.join(gen_outdir, naming)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    needed_dimensions = [(l, w) for l, w in zip(spec["lengths"], spec["widths"])]
    source_dim = (spec["lengths"][-1], spec["widths"][-1])
    target_dim = (spec["lengths"][0], spec["widths"][0])

    morph_class_trained = setup_morpher(source_im, target_im, output_folder=folder,**kwargs)

    log.info(f"Generating {len(needed_dimensions)} images")
    for j, dim in enumerate(needed_dimensions):
        name = f"im_{j:03d}"
        im = single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, name=name, save_images=True)
    # except Exception as e:
    #    log.error(f"Processing of {name_id} FAILED! Trying next! Got error: {e}")

log.info(f"Done, output saved to: {gen_outdir}")
