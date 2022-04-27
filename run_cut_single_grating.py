import os
import sys
import numpy as np
from nilt_base.settingsreader import SetupTool
import morph_tools
from morph_tools import setup_morpher
from grating_helper import get_morphing_info_from_specs
from glob import glob
import argparse
import shutil
import matplotlib.pyplot as plt
import datetime


def fmod(n):
    if n < 5:
        n += 1
    return n


def threshold_image(image, threshold):
    if isinstance(image, str):
        image = morph_tools.load_image(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("Input be a string or numpy ndarray")

    binary = np.zeros_like(image)
    binary[image >= threshold] = 1
    return binary


def split_image_to_blobs(path):
    # Split images into individual blobs
    image = morph_tools.load_image(path)
    im_keypoints = morph_tools.find_blobs(image)

    im_midpoints = [morph_tools.bbox_midpoint(*blob.bbox) for blob in im_keypoints]

    im_blob_files = []
    im_folder = os.path.dirname(path)
    for i, blob in enumerate(im_keypoints):
        blob_file = os.path.join(im_folder, f"blob_{i:02d}.png")
        im_blob_files.append(blob_file)
        morph_tools.save_image(blob_file, blob.image, detect_range=True)

    return im_blob_files, im_midpoints


if __name__ == "__main__":

    st = SetupTool("grating_morpher")
    log = st.log

    parser = argparse.ArgumentParser(description="Run a DiffMorph on a series of optimizations to approximate "
                                                 "intermediate designs.",
                                     add_help=True,
                                     parents=[st.parser]
                                     )

    parser.add_argument('--clean_work_files',
                        action='store_true',
                        help="If intermediate files should be deleted after run.")

    parser.add_argument('-ce', '--compute_efficiency',
                        action='store_true',
                        help="Evaluate performance of gratings.")

    parser.add_argument('--rcwa',
                        type=str,
                        default=os.path.join("..", "RCWA-TopOpt"),
                        help="Path to the RCWA_TopOpt package. Default: ")

    args = parser.parse_args()

    if args.compute_efficiency:
        if not os.path.isdir(args.rcwa):
            raise ModuleNotFoundError(
                f"RCWA-TopOpt module not found at {args.rcwa}, place it there, or provide path using --rwca [PATH].")
        sys.path.append(args.rcwa)
        try:
            import auto_config_generator
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"auto_config_generator.py not found in RCWA-TopOpt folder")

    specification_path = st.read_input("morph.specifications", datatype=str)
    positions_path = st.read_input("morph.positions", datatype=str)
    optimized_dir = st.read_input("morph.optimized_dir", datatype=str)
    image_name = st.read_input("morph.image_name", datatype=str)
    # The "target" identifier, e.g. makes spec N12_1 match folder N12_1_left as the target
    # The "source" identifier, e.g. makes spec N12_1 match folder N12_1_source as the target
    # The target is the top position from the positions_path, and source is the buttom. This naming should reflect that
    to_identifier = st.read_input("morph.to_identifier", datatype=str, default_value="left")
    from_identifier = st.read_input("morph.from_identifier", datatype=str, default_value="right")
    scale = st.read_input("morph.scale", datatype=float)  # resolution of image as um/px
    select_ids = st.read_input("morph.select_ids", datatype=list, default_value=False)
    thresholds = st.read_input("morph.thresholds", datatype=(list, tuple, float, int), default_value=(127.5,))
    if isinstance(thresholds, (float, int)):
        thresholds = (thresholds, )
    log.info("Thresholding image at: " + ", ".join(str(th) for th in thresholds))

    try:
        st.read_input("morph.parameters.im_sz")
        raise ValueError("morph.parameters.im_sz must not be set. It is inferred from specifications")
    except ValueError:
        pass

    parameters = st.settings["morph"].get("parameters", None)

    specs = get_morphing_info_from_specs(specification_path, positions_path, n_mod_fun=fmod)

    # If only certain ID's from the specification file should be morphed
    if select_ids:
        selected = {}
        for sel_id in select_ids:
            selected[sel_id] = specs[sel_id]
        specs = selected
        log.info("DEBUG: Selected IDs:" + ", ".join(select_ids))

    gen_outdir = st.output_folder
    count = 0
    average_efficency = []
    average_efficency_names = []
    for name_id, spec in specs.items():
        count += 1
        log.info(f"Processing {name_id} - {count}/{len(specs)}")
        folders = glob(os.path.join(optimized_dir, name_id + "_*"))
        if not len(folders) == 2:
            raise ValueError(f"Expected to find 2 folders for id {name_id}, but found {len(folders)}")

        target_image = os.path.join(optimized_dir, name_id + f"_{to_identifier}", image_name)
        source_image = os.path.join(optimized_dir, name_id + f"_{from_identifier}", image_name)

        if not os.path.isfile(target_image):
            raise FileNotFoundError(f"Source image, {target_image}, does not exist")
        if not os.path.isfile(source_image):
            raise FileNotFoundError(f"Source image, {source_image}, does not exist")

        # Cutout each individual blob and morph and save it as an image
        trg_blob_files, trg_midpoints = split_image_to_blobs(target_image)
        src_blob_files, src_midpoints = split_image_to_blobs(source_image)

        if not len(src_midpoints) == len(trg_midpoints):
            raise ValueError(f"Different number of Blobs found between source and target. "
                             f"{len(src_midpoints)} vs {len(trg_midpoints)}")

        n_blob = len(src_midpoints)

        # Create general folder
        id_folder = os.path.join(gen_outdir, f"morphed_{name_id}")
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)

        needed_dimensions = [(l, w) for l, w in zip(spec["lengths"], spec["widths"])]
        target_dim = (spec["lengths"][0], spec["widths"][0])
        source_dim = (spec["lengths"][-1], spec["widths"][-1])

        # Create empty images for final morphing
        morphed_images = [np.zeros((int(np.ceil(l / scale)), int(np.ceil(w / scale)), 3)) for l, w in needed_dimensions]
        pct = morph_tools.interpolate_pct(spec["lengths"], source_dim[0], target_dim[0])

        # Go though each blob and morph it
        for i in range(n_blob):
            src_blob = src_blob_files[i]
            trg_blob = trg_blob_files[i]
            blob_folder = os.path.join(id_folder, f"blob_{i:02d}")
            if not os.path.isdir(blob_folder):
                os.makedirs(blob_folder)

            morph_class_trained = setup_morpher(src_blob, trg_blob, output_folder=blob_folder,
                                                padding_args={"extra_pad": 10}, **parameters)

            log.info(f"Generating {len(needed_dimensions)} images")
            for j in range(len(morphed_images)):
                # Get morphing for this dimension and insert into the placeholder image
                im = morph_tools.single_blob_morpher(morph_class_trained, pct[j],
                                                     save_images=(not args.clean_work_files))
                offset_h = int(np.round(src_midpoints[i][0] + (trg_midpoints[i][0] - src_midpoints[i][0]) * pct[j]))
                offset_w = int(np.round(src_midpoints[i][1] + (trg_midpoints[i][1] - src_midpoints[i][1]) * pct[j]))

                h, w = im.shape[0:2]
                vtop, vbot = morph_tools._get_vpos_idx(offset_h * 2, h, "c")
                hleft, hright = morph_tools._get_hpos_idx(offset_w * 2, w, "c")

                morphed_images[j][vtop:vbot, hleft:hright, ...] = im

        # Save the final images
        morphed_fnames = []
        for i, im in enumerate(morphed_images):
            im_name = os.path.join(id_folder,
                                   f'morphed_{i:03d}_{pct[i]:0.2f}pct_'
                                   f'{needed_dimensions[i][0]:0.3f}x{needed_dimensions[i][1]:0.3f}.png')
            morph_tools.save_image(im_name, im)
            morphed_fnames.append(im_name)

        if args.compute_efficiency:
            log.info(f"Computing efficiency of morphings.")

            efficiencies = []

            for name in morphed_fnames:
                eff_th = []
                for th in thresholds:
                    th_image = threshold_image(name, th)
                    th_im_name = os.path.join(os.path.dirname(name), f"th_{th}_" + os.path.basename(name))
                    morph_tools.save_image(th_im_name, th_image)
                    od = auto_config_generator.analyze_image(th_im_name,
                                                             os.path.join(args.rcwa, auto_config_generator.DEFAULT_CONFIG),
                                                             os.path.join(args.rcwa, "configs/obj_modes_1.txt"),
                                                             output_dir=os.path.join(gen_outdir, "efficiency_analysis", name_id, os.path.splitext(os.path.basename(name))[0]))
                    effs, _ = auto_config_generator.compute_efficincy(od,
                                                                      os.path.join(args.rcwa, "configs/obj_modes_1.txt"))
                    eff_th.append(np.mean(effs))
                efficiencies.append(eff_th)
            efficiencies = np.asarray(efficiencies)

            auto_config_generator.pretty_print_array(efficiencies.T*100,
                                                     row_label="Threshold",
                                                     col_label="Morph %",
                                                     row_ticks=thresholds,
                                                     col_ticks=pct*100,
                                                     title=f"{name_id} - Efficiencies, %")

            plt.figure()
            plt.plot(pct, efficiencies * 100, "*-")
            plt.xlabel("Morphed %")
            plt.ylabel("Efficiency %")
            plt.title(f"Efficiency of morphed {name_id}")
            plt.legend([f"Threshold: {th}" for th in thresholds])
            plt.savefig(os.path.join(id_folder, f"efficiency_{name_id}.png"))
            plt.close()

            average_efficency.append(np.mean(efficiencies, axis=0))
            average_efficency_names.append(name_id)

        if args.clean_work_files:
            for i in range(n_blob):
                blob_folder = os.path.join(id_folder, f"blob_{i:02d}")
                shutil.rmtree(blob_folder)
            if args.compute_efficiency:
                shutil.rmtree(os.path.join(gen_outdir, "efficiency_analysis"))

            log.info("DEBUG: Removed temporay files")

    if args.compute_efficiency:
        plt.figure()
        plt.plot(average_efficency_names, np.array(average_efficency) * 100)
        plt.xlabel("Grating IDs")
        plt.ylabel("Average efficiency %")
        plt.title(f"Overall average efficiencies (Date: {datetime.datetime.today().strftime('%Y-%m-%d')})")
        plt.legend([f"Threshold. {th}" for th in thresholds])
        plt.savefig(os.path.join(gen_outdir, f"efficiency_overview.png"))
        plt.close()

    log.info(f"Morphing done. Output saved to: {gen_outdir}")
