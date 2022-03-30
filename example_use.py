import os
import numpy as np
import cv2
import morph_tools
from morph_tools import morph, crop_image_to_size, pad_image_to_square, load_image, save_image, \
    pad_images_to_same_square

import grating_helper



"""
Single feature at a time morph
"""

org = load_image("data/x_p_1.625_1nm (Phone).png")
trg = load_image("data/x_p_1.925_1nm (Phone).png")

org_keypoints = morph_tools.find_blobs(org)
trg_keypoints = morph_tools.find_blobs(trg)


print(org_keypoints)
print(trg_keypoints)

blob_dir = "data/blobs"
try:
    os.mkdir(blob_dir)
except FileExistsError:
    pass

for i, blob in enumerate(org_keypoints):
    blob_file = os.path.join(blob_dir, f"org_blob_{i:02d}.png")
    save_image(blob_file, blob.image, detect_range=True)
for i, blob in enumerate(trg_keypoints):
    blob_file = os.path.join(blob_dir, f"trg_blob_{i:02d}.png")
    save_image(blob_file, blob.image, detect_range=True)





"""
Expample of how to use the functions in morph_tools, to load two images, pad them to be square,
morph between them and finialy crop the morhped images to a given size.


# Load images
org = load_image("data/from_11.png")
trg = load_image("data/to_11.png")

# Remeber their original dimensions
from_h = org.shape[0]
to_h = trg.shape[0]
width = trg.shape[1]

# Pad them to the same square size
org, trg = pad_images_to_same_square(org, trg, color="black")

# Save them so the path can be passed to morph()
org_name_padded = "data/from_11_padded.png"
trg_name_padded = "data/to_11_padded.png"
save_image(org_name_padded, org)
save_image(trg_name_padded, trg)

# Train and morph the images
png_image_paths, npy_image_paths = morph(org_name_padded,  # Source file name
                                         trg_name_padded,  # Target file name
                                         10,  #
                                         output_folder="morph/small_test/",
                                         im_sz=org.shape[0],
                                         mp_sz=200,
                                         train_epochs=50,
                                         )

# Just linear interpolated heights for now
heights = np.linspace(from_h, to_h, len(png_image_paths))
final_dir = "morph/small_test/final"
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

for i, p in enumerate(png_image_paths):
    im = load_image(p)
    im = crop_image_to_size(im, (heights[i], width), scale=1, pos="cc")
    save_image(os.path.join(final_dir, f"final_{i:05d}.png"), im)

print("Done")
"""


## New example


"""
# Psudo example 
specs = grating_helper.get_morphing_info_from_specs(specification_path, positions_path)

i = 0
scale = 0.005 # Scale um/px for this run

for source, target in zip(source_images, target_images):
    morph_class_trained = morph_tools.setup_morpher(source, target, output_folder=f"Outputs_{i}")

    # Get relevant specs and dimensions for with specs
    ....

    source_dim = source.shape[:2]
    target_dim = target.shape[:2]
    for dim in needed_dimensions:
        im = morph_tools.single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, save_images=True)

print("Done")


import os
import numpy as np
import morph_tools
from morph_tools import morph, crop_image_to_size, pad_image_to_square, load_image, save_image, \
    pad_images_to_same_square

import grating_helper
from glob import glob

specification_path = "data/grating_design_specifications.txt"
positions_path = "data/grating_position.txt"

specs = grating_helper.get_morphing_info_from_specs(specification_path, positions_path)

source_images = glob("")

for source, target in zip(source_images, target_images):
    morph_class_trained = morph_tools.setup_morpher(source, target, output_folder=f"Outputs_{i}")

    # Get relevant specs and dimensions for with specs
    ....

    source_dim = source.shape[:2]
    target_dim = target.shape[:2]
    for dim in needed_dimensions:
        im = morph_tools.single_image_morpher(morph_class_trained, dim, source_dim, target_dim, scale, save_images=True)



## Fix specification file
"""
te = 1000
for i in range(1000):
    epoch = i + 1
    if epoch == 1:
        print(f"Training ({te} epochs): ", end=" ")


    if (epoch < 100 and epoch % 10 == 0) or \
            (epoch < 1000 and epoch % 100 == 0) or \
            (epoch % 1000 == 0):
        # "Replace" tqdm
        print(f"{epoch / te * 100:.1f}", end=" ")

    time.sleep(0.01)

print("Done Training!")


print("Next thing")