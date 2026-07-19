import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os
from datetime import datetime

# ─── STYLE CONFIGURATION ───────────────────────────────────────────────────────
COLORS = {
    "car":        "#00BFFF",   # vivid sky blue
    "truck":      "#FF9F1C",   # amber orange
    "bus":        "#2ECC71",   # emerald green
    "motorbike":  "#FF0099",   # hot pink / magenta
    "tricycle":   "#FFD600",   # golden yellow
    "van":        "#9B59B6",   # soft lavender purple
    "IN":         "#00BCD4",   # cyan
    "OUT":        "#FF5722",   # deep orange
}

BACKGROUND   = "#0F1923"   # dark navy
PANEL        = "#1A2535"   # slightly lighter panel
GRID_COLOR   = "#263548"
TEXT_COLOR   = "#E8EDF2"
ACCENT       = "#00BCD4"

def apply_dark_style(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark professional styling to an axes."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=TEXT_COLOR, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_COLOR, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR, labelpad=6)


def generate_graphs(csv_path=None, output_dir=None):
    # ✅ Default to paths anchored to THIS file's own folder, not the
    # process's current working directory — this is what caused output to
    # silently go missing depending on how the app was launched.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(base_dir, "output.csv")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "graphs")

    if not os.path.exists(csv_path):
        print("No data file found!")
        return "no_file"

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path)
    if len(data) == 0:
        print("No data to plot!")
        return "empty"

    data["Time"] = pd.to_datetime(data["Time"])
    data["Speed(km/h)"] = pd.to_numeric(data["Speed(km/h)"], errors="coerce").fillna(0)

    vehicle_types = data["Vehicle"].unique().tolist()
    total_vehicles = len(data)
    avg_speed = data["Speed(km/h)"].mean()
    total_in  = len(data[data["Direction"] == "IN"])
    total_out = len(data[data["Direction"] == "OUT"])

    # ── FIGURE 1: MAIN DASHBOARD ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BACKGROUND)
    fig.suptitle(
        "VEHICLE DETECTION & COUNTING SYSTEM — TRAFFIC ANALYSIS REPORT",
        fontsize=14, fontweight="bold", color=ACCENT,
        y=0.97, fontfamily="DejaVu Sans"
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.91, bottom=0.08, left=0.07, right=0.97
    )

    # ── PLOT 1: Vehicle Type Count (Bar) ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = data["Vehicle"].value_counts()
    bar_colors  = [COLORS.get(v, "#9E9E9E") for v in type_counts.index]
    bars = ax1.bar(type_counts.index, type_counts.values,
                   color=bar_colors, edgecolor=BACKGROUND, linewidth=0.8,
                   width=0.55, zorder=3)
    for bar, val in zip(bars, type_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=9, fontweight="bold")
    apply_dark_style(ax1,
                     title="Vehicle Type Distribution",
                     xlabel="Vehicle Type",
                     ylabel="Count")
    ax1.set_xticks(range(len(type_counts)))
    ax1.set_xticklabels(type_counts.index, rotation=15, ha="right")

    # ── PLOT 2: IN vs OUT per Vehicle Type (Grouped Bar) ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    direction_counts = data.groupby(["Vehicle", "Direction"]).size().unstack(fill_value=0)
    x      = np.arange(len(direction_counts))
    width  = 0.35
    has_in  = "IN"  in direction_counts.columns
    has_out = "OUT" in direction_counts.columns
    if has_in:
        b1 = ax2.bar(x - width/2, direction_counts["IN"],  width,
                     label="IN",  color=COLORS["IN"],  edgecolor=BACKGROUND,
                     linewidth=0.8, zorder=3)
    if has_out:
        b2 = ax2.bar(x + width/2, direction_counts["OUT"], width,
                     label="OUT", color=COLORS["OUT"], edgecolor=BACKGROUND,
                     linewidth=0.8, zorder=3)
    apply_dark_style(ax2,
                     title="IN vs OUT by Vehicle Type",
                     xlabel="Vehicle Type",
                     ylabel="Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(direction_counts.index, rotation=15, ha="right")
    ax2.legend(facecolor=PANEL, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8)

    # ── PLOT 3: Speed Distribution (Histogram) ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    speed_data = data[data["Speed(km/h)"] > 0]["Speed(km/h)"]
    if len(speed_data) > 0:
        n, bins, patches = ax3.hist(speed_data, bins=15,
                                    color=ACCENT, edgecolor=BACKGROUND,
                                    linewidth=0.6, alpha=0.85, zorder=3)
        # color gradient on bars
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(
                plt.cm.cool(left / (speed_data.max() + 1))
            )
        ax3.axvline(speed_data.mean(), color="#FFD700", linewidth=1.5,
                    linestyle="--", label=f"Mean: {speed_data.mean():.1f} km/h")
        ax3.legend(facecolor=PANEL, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=8)
    apply_dark_style(ax3,
                     title="Speed Distribution",
                     xlabel="Speed (km/h)",
                     ylabel="Frequency")

    # ── PLOT 4: Traffic Flow Over Time (Line) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    data_time = data.set_index("Time")
    flow = data_time.resample("1min").size()
    ax4.fill_between(flow.index, flow.values,
                     color=ACCENT, alpha=0.15, zorder=2)
    ax4.plot(flow.index, flow.values,
             color=ACCENT, linewidth=2, marker="o",
             markersize=4, markerfacecolor=ACCENT,
             markeredgecolor=BACKGROUND, zorder=3)
    # per vehicle type overlay
    for vtype in vehicle_types:
        vflow = data_time[data_time["Vehicle"] == vtype].resample("1min").size()
        if len(vflow) > 0:
            ax4.plot(vflow.index, vflow.values,
                     color=COLORS.get(vtype, "#9E9E9E"),
                     linewidth=1.2, linestyle="--", alpha=0.7,
                     label=vtype)
    apply_dark_style(ax4,
                     title="Traffic Flow Over Time (vehicles per minute)",
                     xlabel="Time",
                     ylabel="Vehicle Count")
    ax4.legend(facecolor=PANEL, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8, loc="upper right")
    ax4.tick_params(axis="x", rotation=20)

    # ── PLOT 5: Speed per Vehicle Type (Box Plot) ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    speed_by_type = [
        data[data["Vehicle"] == v]["Speed(km/h)"].values
        for v in vehicle_types
    ]
    speed_by_type = [s[s > 0] for s in speed_by_type]
    if any(len(s) > 0 for s in speed_by_type):
        bp = ax5.boxplot(speed_by_type, patch_artist=True,
                         medianprops=dict(color="#FFD700", linewidth=2),
                         whiskerprops=dict(color=TEXT_COLOR),
                         capprops=dict(color=TEXT_COLOR),
                         flierprops=dict(marker="o", color=ACCENT,
                                         markersize=3, alpha=0.5))
        for patch, vtype in zip(bp["boxes"], vehicle_types):
            patch.set_facecolor(COLORS.get(vtype, "#9E9E9E"))
            patch.set_alpha(0.8)
        ax5.set_xticks(range(1, len(vehicle_types) + 1))
        ax5.set_xticklabels(vehicle_types, rotation=15, ha="right")
    apply_dark_style(ax5,
                     title="Speed Distribution by Vehicle Type",
                     xlabel="Vehicle Type",
                     ylabel="Speed (km/h)")

    # ── PLOT 6: Summary Statistics Panel ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    for spine in ax6.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.5)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title("Summary Statistics",
                  fontsize=11, fontweight="bold",
                  color=TEXT_COLOR, pad=10)

    stats = [
        ("Total Vehicles",    str(total_vehicles)),
        ("Vehicles IN",       str(total_in)),
        ("Vehicles OUT",      str(total_out)),
        ("Avg Speed",         f"{avg_speed:.1f} km/h"),
        ("Vehicle Types",     str(len(vehicle_types))),
        ("Duration",          f"{(data['Time'].max()-data['Time'].min()).seconds//60} min"),
    ]
    for i, (label, value) in enumerate(stats):
        y_pos = 0.88 - i * 0.145
        ax6.text(0.08, y_pos, label + ":",
                 transform=ax6.transAxes,
                 fontsize=9, color="#90A4AE", va="center")
        ax6.text(0.92, y_pos, value,
                 transform=ax6.transAxes,
                 fontsize=10, fontweight="bold",
                 color=ACCENT, va="center", ha="right")
        if i < len(stats) - 1:
            ax6.plot([0.05, 0.95], [y_pos - 0.06, y_pos - 0.06],
                     color=GRID_COLOR, linewidth=0.5,
                     transform=ax6.transAxes, clip_on=False)

    # ── TIMESTAMP FOOTER ──────────────────────────────────────────────────────
    fig.text(0.99, 0.01,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha="right", va="bottom",
             fontsize=7, color="#546E7A")

    plt.savefig(os.path.join(output_dir, "traffic_analysis_dashboard.png"),
                dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    print("✅ Dashboard saved → graphs/traffic_analysis_dashboard.png")

    # ── FIGURE 2: PRECISION / ACCURACY REPORT ────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BACKGROUND)
    fig2.suptitle("DETECTION ACCURACY REPORT",
                  fontsize=13, fontweight="bold",
                  color=ACCENT, y=1.01)

    # Pie chart — vehicle type share
    ax_pie = axes[0]
    ax_pie.set_facecolor(PANEL)
    pie_colors = [COLORS.get(v, "#9E9E9E") for v in type_counts.index]
    wedges, texts, autotexts = ax_pie.pie(
        type_counts.values,
        labels=type_counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(edgecolor=BACKGROUND, linewidth=1.5),
        textprops=dict(color=TEXT_COLOR, fontsize=9)
    )
    for at in autotexts:
        at.set_color(BACKGROUND)
        at.set_fontweight("bold")
        at.set_fontsize(8)
    ax_pie.set_title("Vehicle Type Share",
                     fontsize=11, fontweight="bold",
                     color=TEXT_COLOR, pad=12)

    # Horizontal bar — direction count per vehicle
    ax_hbar = axes[1]
    ax_hbar.set_facecolor(PANEL)
    if has_in and has_out:
        y_pos = np.arange(len(direction_counts))
        ax_hbar.barh(y_pos - 0.2, direction_counts["IN"],  0.35,
                     color=COLORS["IN"],  label="IN",
                     edgecolor=BACKGROUND, linewidth=0.8)
        ax_hbar.barh(y_pos + 0.2, direction_counts["OUT"], 0.35,
                     color=COLORS["OUT"], label="OUT",
                     edgecolor=BACKGROUND, linewidth=0.8)
        ax_hbar.set_yticks(y_pos)
        ax_hbar.set_yticklabels(direction_counts.index)
        ax_hbar.legend(facecolor=PANEL, edgecolor=GRID_COLOR,
                       labelcolor=TEXT_COLOR, fontsize=8)
    apply_dark_style(ax_hbar,
                     title="IN vs OUT Count per Vehicle Type",
                     xlabel="Count",
                     ylabel="Vehicle Type")

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(output_dir, "accuracy_report.png"),
                dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    print("✅ Accuracy report saved → graphs/accuracy_report.png")
    plt.close("all")
    print("\n📊 All graphs saved in 'graphs/' folder")
    return "ok"
