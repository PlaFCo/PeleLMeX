import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

def parse_log(filename):
    times, p_coords, pabs_coords = [], [], []
    curr_t, curr_sdc = None, 0
    t_pat = re.compile(r"STEP\s*\[\d+\]\s*-\s*Time:\s*([0-9.eE+-]+)")
    sdc_pat = re.compile(r"SDC iter\s*\[(\d+)\]")
    c_pat = re.compile(r"Plasma center\s*=\s*\(\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\s*\)")
    p_pat = re.compile(r"p_abs center\s*=\s*\(\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\s*\)")
    temp_p, temp_pabs = None, None
    with open(filename, 'r') as f:
        for line in f:
            if "STEP" in line: curr_sdc = 0
            if (t := t_pat.search(line)): curr_t = float(t.group(1))
            if (s := sdc_pat.search(line)): curr_sdc = int(s.group(1))
            if (c := c_pat.search(line)) and curr_t is not None and curr_sdc != 2: temp_p = (float(c.group(1)), float(c.group(2)))
            if (p := p_pat.search(line)) and curr_t is not None and curr_sdc != 2: temp_pabs = (float(p.group(1)), float(p.group(2)))
            if temp_p is not None and temp_pabs is not None:
                times.append(curr_t); p_coords.append(temp_p); pabs_coords.append(temp_pabs)
                temp_p, temp_pabs = None, None
    return np.array(times), np.array(p_coords), np.array(pabs_coords)

def plot_data(times, p_xy, pabs_xy):
    if len(times) > 5:
        times, p_xy, pabs_xy = times[5:], p_xy[5:] * 1000.0, pabs_xy[5:] * 1000.0
    else:
        p_xy, pabs_xy = p_xy * 1000.0, pabs_xy * 1000.0
    pr, p_theta = np.sqrt(p_xy[:,0]**2 + p_xy[:,1]**2), np.degrees(np.arctan2(p_xy[:,1], p_xy[:,0]))
    pabs_r, pabs_theta = np.sqrt(pabs_xy[:,0]**2 + pabs_xy[:,1]**2), np.degrees(np.arctan2(pabs_xy[:,1], pabs_xy[:,0]))
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.scatter(p_xy[:,0], p_xy[:,1], c=times, cmap='viridis', s=60, marker='o', label='Plasma Center', edgecolor='k', linewidth=0.4)
    ax1.scatter(pabs_xy[:,0], pabs_xy[:,1], c=times, cmap='plasma', s=60, marker='^', label='p_abs Center', edgecolor='k', linewidth=0.4)
    cb = fig1.colorbar(ax1.collections[0])
    cb.set_label('Time (s)', fontsize=16)
    ax1.set_xlim(-3.0, 3.0); ax1.set_ylim(-3.0, 3.0)
    ax1.set_xlabel('X Position (mm)'); ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Plasma Center vs p_abs Center Trajectory', pad=15)
    ax1.legend(loc='best'); ax1.grid(True, linestyle='--', alpha=0.6)
    fig1.tight_layout(); fig1.savefig("analysis_outputs/plasma_and_pabs_xy.png", dpi=300)
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(times, pr, color='tab:blue', linewidth=2, label='Plasma Radius')
    ax2.plot(times, pabs_r, color='tab:blue', linestyle='--', linewidth=2, label='p_abs Radius')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Radius (mm)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax3 = ax2.twinx()
    ax3.plot(times, p_theta, color='tab:orange', linewidth=2, label='Plasma Angle')
    ax3.plot(times, pabs_theta, color='tab:orange', linestyle='--', linewidth=2, label='p_abs Angle')
    ax3.set_ylabel('Angle (degrees)', color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_title('Radius and Angle vs Time', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig2.tight_layout(); fig2.savefig("analysis_outputs/plasma_centers_time_series.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    base_dir = "../../Exec/Production/3D_Microwave"
    t, p_xy, pabs_xy = parse_log(base_dir + "/3D_N2O2-1613.out")
    if len(t) > 0: plot_data(t, p_xy, pabs_xy)