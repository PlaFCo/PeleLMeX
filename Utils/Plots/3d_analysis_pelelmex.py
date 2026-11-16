import os
from pathlib import Path
import re
import numpy as np
import yt

base_dir = "../../Exec/Production/3D_Microwave/outdir"
slice_axes = ["x","y", "z"]         # ["x","y", "z"]

z_slice_center = 0.281
use_all_plt = False

cfd_plots_config = {
    "extsource_rhoh": {"log_scale": False, "fix_scale": False},
    "x_velocity": {"log_scale": False, "fix_scale": True},
    "y_velocity": {"log_scale": False, "fix_scale": False},
    "z_velocity": {"log_scale": False, "fix_scale": False},
    "mag_vel": {"log_scale": False, "fix_scale": False},
    "temp": {"log_scale": False, "fix_scale": False},
    "X(O2)": {"log_scale": False, "fix_scale": False},
    "X(N2)": {"log_scale": False, "fix_scale": False},
    "X(NO)": {"log_scale": False, "fix_scale": False},
    "divu": {"log_scale": False, "fix_scale": False},
}

harps_plots_config = {
    "electron_temp": {"log_scale": False, "fix_scale": False},
    "E_field": {"log_scale": False, "fix_scale": False},
    "E_over_N": {"log_scale": False, "fix_scale": False},
    "electron_dens": {"log_scale": False, "fix_scale": False},
    #"te_over_tg": {"log_scale": False, "fix_scale": False},
}

z_limits = {"temp": (275, 6050), "z_velocity": (-5, 5), "y_velocity": (-5.0, 5.0), "x_velocity": (-5.0, 5.0), "extsource_rhoh": (1e-4, 0.5)}

def get_all_plt_dirs(base_dir, pattern="plt_N2_*"):
    return sorted([str(p) for p in Path(base_dir).glob(pattern) if p.is_dir() and re.search(r"(\d+)$", p.name)], key=lambda p: int(re.search(r"(\d+)$", Path(p).name).group(1)))


def get_colormap(quantity):
    if any(k in quantity for k in ("Y(", "dens", "X(")): return "viridis"
    elif "velocity" in quantity: return "RdBu_r"
    elif "extsource" in quantity: return "inferno"
    elif "temp" in quantity: return "magma"
    elif "E_over_N" in quantity: return "plasma"
    return "CMRmap"

def generate_averaged_slice_plots(plt_folders, config, axes, prefix="3d_slice_avg"):
    if not plt_folders: print(f"[{prefix}] Error: No plotfiles found."); return

    sample_ds = yt.load(plt_folders[0], units_override={"length_unit": (1.0, "m")})
    available_fields = [f[1] for f in sample_ds.field_list] + [f[1] for f in sample_ds.derived_field_list]

    for slice_axis in axes:
        for quantity, settings in config.items():
            if quantity not in available_fields: continue

            print(f"[{prefix}] Averaging {quantity} along {slice_axis}-axis across {len(plt_folders)} frames...")
            frb_arrays, last_slc = [], None

            for folder in plt_folders:
                ds = yt.load(folder, units_override={"length_unit": (1.0, "m")})
                slc = yt.SlicePlot(ds, slice_axis, quantity, center=[0.0, 0.0, z_slice_center]) if slice_axis == "z" else yt.SlicePlot(ds, slice_axis, quantity, origin="native")
                frb_arrays.append(np.array(slc.frb[quantity]))
                last_slc = slc

            avg_values = np.stack(frb_arrays).mean(axis=0)
            log_scale, fix_scale = settings["log_scale"], settings["fix_scale"]

            if fix_scale and quantity in z_limits:
                z_min, z_max = z_limits[quantity]
                if log_scale and z_min <= 0: z_min = 1e-6
            else:
                z_min, z_max = float(np.nanmin(avg_values)), float(np.nanmax(avg_values))
                if log_scale and z_min <= 0: z_min = 1e-6

            if log_scale and not fix_scale: last_slc.set_log(quantity, True, linthresh=1e-6)
            else: last_slc.set_log(quantity, log_scale)

            last_slc.set_zlim(quantity, z_min, z_max)
            last_slc.set_cmap(quantity, get_colormap(quantity))
            last_slc.set_figure_size(7)
            last_slc.set_font({"size": 14, "family": "serif"})
            last_slc._setup_plots()
            last_slc.plots[quantity].image.set_data(avg_values)

            # Increase axis labels and colorbar font sizes
            ax = last_slc.plots[quantity].axes
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            cb = last_slc.plots[quantity].cb
            cb.ax.yaxis.label.set_fontsize(18)
            cb.ax.tick_params(labelsize=14)

            out_name = f"analysis_outputs/{prefix}_{slice_axis}_{quantity}.png"
            last_slc.save(out_name)
            print(f"[{prefix}] Saved: {out_name}")

if __name__ == "__main__":
    if use_all_plt:
        plt_folders = get_all_plt_dirs(base_dir, "plt_N2_*")
        plt_folders_harps = get_all_plt_dirs(base_dir, "plt_harps_N2_*")
    else:
        base_step, step_size, n_folders = 80500, 500, 1
        base_step_harps, step_size_harps, n_folders_harps = 500, 500, 1

        steps = [base_step - i * step_size for i in range(n_folders)]

        plt_folders = [f"{base_dir}/plt_N2_{step}" for step in steps]
        steps_harps = [base_step_harps - i * step_size_harps for i in range(n_folders_harps)]
        plt_folders_harps = [f"{base_dir}/plt_harps_N2_00{step}" for step in steps_harps]

    if plt_folders:
        generate_averaged_slice_plots(plt_folders, cfd_plots_config, axes=slice_axes, prefix="3d_slice_avg")

    if plt_folders_harps:
        generate_averaged_slice_plots(plt_folders_harps, harps_plots_config, axes=slice_axes, prefix="harps_slice_avg") 