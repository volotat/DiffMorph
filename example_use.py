import os
import numpy as np
from morph_tools import morph, crop_image_to_size, pad_image_to_square, load_image, save_image, pad_images_to_same_square

"""
Expample of how to use the functions in morph_tools, to load two images, pad them to be square,
morph between them and finialy crop the morhped images to a given size.

"""

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
                                         10,               #
                                         output_folder="morph/small_test/",
                                         im_sz=org.shape[0],
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

