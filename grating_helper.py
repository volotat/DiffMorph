import numpy as np
import pandas as pd


def move_right_on_nan(row):
    """ Used for aligning rows to the right if nan is detected.
        The way Grating Design Specifications is structured, result in pandas reading it as the NaN's are
        in the last column, instead of the first. Applying this method fixes that.
    """
    nans = np.isnan(row)
    if np.any(nans):
        nan_values = row[nans].values
        real_values = row[np.invert(nans)]
        row[len(nan_values):] = real_values
        row[:len(nan_values)] = nan_values
    return row


def get_morphing_info_from_specs(specification_path, positions_path, n_mod_fun=None):
    """ Parse corresponding specification and positions, and get number of inbetween needed morphing and
        their lengths and widths

    Parameters
    ----------
    specification_path : str
        Path to the specification file for the grating, like: grating_design_specifications.txt"

    positions_path : str
        Path to the Grating position file, like: grating_position.txt

    n_mod_fun : function
        Function that modify the N value if needed. Default: None

    Returns
    -------

    """
    # Read specifications
    df_specs = pd.read_csv(specification_path, comment="#", sep='\s{3,}', engine="python")
    df_specs.apply(move_right_on_nan, axis=1)  # # Fix nan position

    # Read optimized positions
    df_pos = pd.read_csv(positions_path, comment="#", engine="python")
    # for i in range()

    # Extract the correposning collumn names
    spec_n_col = [name for name in df_specs.columns if "N" in name][0]
    spec_length_col = [name for name in df_specs.columns if "length" in name.lower()][0]
    spec_width_col = [name for name in df_specs.columns if "width" in name.lower()][0]

    pos_n_col = [name for name in df_pos.columns if "N" in name][0]
    pos_length_col = [name for name in df_pos.columns if "length" in name.lower()][0]
    pos_width_col = [name for name in df_pos.columns if "width" in name.lower()][0]

    # Get info for each morphing
    morhings = {}
    id_count = 0
    for i in np.arange(0, len(df_specs), 2):
        id_count += 1
        specs = df_specs[i:i + 2]
        N = int(np.max(specs[spec_n_col]))
        min_len = specs[spec_length_col].min()
        max_len = specs[spec_length_col].max()
        pos_this_n = df_pos.loc[df_pos[pos_n_col] == N]
        pos = pos_this_n.loc[(pos_this_n[pos_length_col] >= min_len) & (pos_this_n[pos_length_col] <= max_len)]

        lengths = pos[pos_length_col].values
        widths = pos[pos_width_col].values

        if n_mod_fun:
            N = n_mod_fun(N)
        name_id = f"N{N}_{id_count}"
        morhings[name_id] = {"N": N,
                             "steps": len(pos),
                             "lengths": lengths,
                             "widths": widths,
                             }

    return morhings


def pretty_print(morhping_dict):
    for key, item in morhping_dict.items():
        print(f"ID: {key}")
        for k, v in item.items():
            print(f"{k:>15s}", ":", v)


if __name__ == "__main__":
    spec_path = "data/grating_design_specifications_test.txt"
    pos_path = "data/grating_position_test.txt"

    morhings_needed = get_morphing_info_from_specs(spec_path, pos_path)

    pretty_print(morhings_needed)
