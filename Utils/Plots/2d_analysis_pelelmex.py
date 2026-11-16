import os
import yt
import numpy as np

data_dir     = "../../Exec/Production/2DSwirlCase"
cfd_prefix   = "outdir/plt_N2_10slm_600W_reg_"
harps_prefix = "outdir/plt_N2_10slm_600W_no_use"

start_step  = 15957     # last 
step_stride = 500     # go backwards in steps of this size
n_datasets  = 1

steps = [start_step - i * step_stride for i in range(n_datasets)]


cfd_plots_config = {
    "temp": {"log_scale": False, "fix_scale": False},
    "y_velocity": {"log_scale": False, "fix_scale": True},
    #"x_velocity": {"log_scale": False, "fix_scale": True},
    #"AngMom": {"log_scale": True, "fix_scale": False},
    "extsource_rhoh_W_cm3": {"log_scale": False, "fix_scale": False},
    #"Y(O)": {"log_scale": False, "fix_scale": False},
    "Y(N)": {"log_scale": False, "fix_scale": False},
    #"Y(NO)": {"log_scale": False, "fix_scale": False},
    #"Y(NO2)": {"log_scale": False, "fix_scale": False},
    #"Y(N2)": {"log_scale": False, "fix_scale": False},
    #"Y(O2)": {"log_scale": False, "fix_scale": False},
    "density": {"log_scale": False, "fix_scale": False},
    #"RhoRT": {"log_scale": False, "fix_scale": False},
    "ang_velocity": {"log_scale": False, "fix_scale": False},
}

harps_plots_config = {
    "electron_temp": {"log_scale": False, "fix_scale": False},
    "E_over_N": {"log_scale": False, "fix_scale": False},
    "electron_dens": {"log_scale": False, "fix_scale": False},
    "te_over_tg": {"log_scale": False, "fix_scale": False},
}

legend_labels = {
    "temp": r"$T_g\ [\mathrm{K}]$",
    "ang_velocity": r"$u_\theta\ [\mathrm{m/s}]$",
    "y_velocity": r"$u_z\ [\mathrm{m/s}]$",
    "x_velocity": r"$u_r\ [\mathrm{m/s}]$",
    "AngMom": r"$l_z\ [\mathrm{kg\ m^2/s}]$",
    "extsource_rhoh_W_cm3": r"$p_\mathrm{abs}\ [\mathrm{W/cm^3}]$",
    "electron_temp": r"$T_e\ [\mathrm{K}]$",
    "E_over_N": r"$E/N\ [\mathrm{Td}]$",
    "electron_dens": r"$n_e\ [\mathrm{m^{-3}}]$",
}

z_limits = {
    "temp": (275, 6050),
    "z_velocity": (-2.5, 2.5),
    "y_velocity": (-5, 5),
    "x_velocity": (-2, 2),
    "AngMom": (0.0, 0.03),
    "avg_pressure": (-0.15, 0.15),
    "extsource_rhoh_W_cm3": (0, 130),
    "electron_temp": (0, 6000),
    "E_over_N": (0, 10),
    "ang_velocity": (0, 10),
}


def _ang_velocity(field, data):
    r_safe = np.maximum(data["index", "r"], data.ds.quan(1e-12, "m"))
    return 1e-4 * data["AngMom"] / (data["density"] * r_safe)

def _extsource_rhoh_W_cm3(field, data):
    return data["boxlib", "extsource_rhoh"] / 1e6

def register_cfd_fields(ds):
    ds.add_field(name=("gas", "ang_velocity"), function=_ang_velocity,
                 sampling_type="cell", units="cm**2/g", force_override=True)
    ds.add_field(name=("gas", "extsource_rhoh_W_cm3"), function=_extsource_rhoh_W_cm3,
                 sampling_type="cell", units="auto", force_override=True)


def load_series(prefix, register=None):
    dss = []
    for s in steps:
        path = os.path.join(data_dir, f"{prefix}{s}")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        ds = yt.load(path, units_override={"length_unit": (1.0, "m")})
        if register is not None:
            register(ds)
        dss.append(ds)
    print(f"Loaded {len(dss)}/{len(steps)} datasets for prefix '{prefix}'")
    return dss

def average_fields(dss, names):
    level = max(ds.index.max_level for ds in dss)
    sums, units = {}, {}

    for i, ds in enumerate(dss, start=1):
        print(f"  [{i}/{len(dss)}] {ds}")
        dims = ds.domain_dimensions.copy()
        dims[: ds.dimensionality] *= ds.relative_refinement(0, level)
        cg = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=dims)

        for name in names:
            arr = cg[name]
            if name not in sums:
                sums[name] = np.zeros(arr.shape, dtype=np.float64)
                units[name] = str(arr.units)
            sums[name] += arr.to(units[name]).d

    return {n: s / len(dss) for n, s in sums.items()}, units

def make_averaged_dataset(ref_ds, arrays, units):
    geom = getattr(ref_ds.geometry, "value", ref_ds.geometry)
    bbox = np.array([ref_ds.domain_left_edge.d, ref_ds.domain_right_edge.d]).T
    data = {name: (arr, units[name]) for name, arr in arrays.items()}
    shape = next(iter(arrays.values())).shape
    return yt.load_uniform_grid(data, shape, length_unit="m", bbox=bbox, geometry=(str(geom), tuple(ref_ds.coordinates.axis_order)), periodicity=tuple(ref_ds.periodicity),)


def generate_slice_plots(ds, config, prefix):
    for quantity, settings in config.items():
        print(f"[{prefix}] Processing {quantity}...")
        log_scale, fix_scale = settings["log_scale"], settings["fix_scale"]

        if "Y(" in quantity or "dens" in quantity:
            cmap = "viridis"
        elif "AngMom" in quantity or "E_over_N" in quantity:
            cmap = "plasma"
        elif "velocity" in quantity:
            cmap = "RdBu_r"
        elif "extsource" in quantity:
            cmap = "inferno"
        elif "temp" in quantity:
            cmap = "magma"
        else:
            cmap = "CMRmap"

        slc = yt.SlicePlot(ds, "theta", quantity, origin="native")
        slc.swap_axes()

        if log_scale and not fix_scale:
            slc.set_log(quantity, True, linthresh=1e-6)
        else:
            slc.set_log(quantity, log_scale)

        if fix_scale and quantity in z_limits:
            z_min, z_max = z_limits[quantity]
            if log_scale and z_min <= 0:
                z_min = 1e-6
            slc.set_zlim(quantity, z_min, z_max)

        if quantity in legend_labels:
            slc.set_colorbar_label(quantity, legend_labels[quantity])

        slc.set_cmap(quantity, cmap)
        slc.set_font({"size": 12, "family": "serif"})

        out_name = f"analysis_outputs/2d_{prefix}_{quantity}.png"
        slc.save(out_name, mpl_kwargs={"bbox_inches": "tight", "pad_inches": 0.05})
        print(f"Saved: {out_name}")


print(f"Averaging snapshots {steps[0]} -> {steps[-1]} (stride {step_stride}, N={n_datasets})")

# ---- CFD ----
cfd_dss = load_series(cfd_prefix, register=register_cfd_fields)
cfd_avg, cfd_units, ds_cfd_avg = None, None, None

if cfd_dss:
    cfd_fields = list(cfd_plots_config)
    need_te_over_tg = "te_over_tg" in harps_plots_config
    if need_te_over_tg and "temp" not in cfd_fields:
        cfd_fields.append("temp")          # needed for Te/Tg

    print("Averaging CFD fields...")
    cfd_avg, cfd_units = average_fields(cfd_dss, cfd_fields)
    np.savez(f"avg_cfd_{len(cfd_dss)}snaps.npz", **cfd_avg)

    ds_cfd_avg = make_averaged_dataset(cfd_dss[0], cfd_avg, cfd_units)
    generate_slice_plots(ds_cfd_avg, cfd_plots_config, prefix="cfd_avg")

# ---- HARPS ----
harps_dss = load_series(harps_prefix)

if harps_dss:
    harps_fields = [k for k in harps_plots_config if k != "te_over_tg"]

    print("Averaging HARPS fields...")
    harps_avg, harps_units = average_fields(harps_dss, harps_fields)

    plot_config = dict(harps_plots_config)
    if "te_over_tg" in plot_config:
        # ratio of the two averaged fields: <Te> / <Tg>
        if cfd_avg is not None and cfd_avg["temp"].shape == harps_avg["electron_temp"].shape:
            harps_avg["te_over_tg"] = harps_avg["electron_temp"] / cfd_avg["temp"]
            harps_units["te_over_tg"] = "dimensionless"
        else:
            print("Warning: cannot compute 'te_over_tg' (no CFD data or grid mismatch)")
            plot_config.pop("te_over_tg")

    np.savez(f"avg_harps_{len(harps_dss)}snaps.npz", **harps_avg)

    ds_harps_avg = make_averaged_dataset(harps_dss[0], harps_avg, harps_units)
    generate_slice_plots(ds_harps_avg, plot_config, prefix="harps_avg")