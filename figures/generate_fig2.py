"""Generate Figure 2: SOPBench pass rate trajectory across 3 outer loops."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Actual data from main_sop_bank.json
outer1 = [0.35, 0.40, 0.50, 0.40, 0.50, 0.55, 0.45, 0.55]
outer2 = [0.50, 0.60, 0.60, 0.60, 0.60, 0.45, 0.60, 0.60]
outer3 = [0.55, 0.65, 0.65, 0.70, 0.55, 0.45, 0.50, 0.65]
iters = np.arange(1, 9)

fig, ax = plt.subplots(figsize=(4.5, 2.8))

# Colors: blue for O1, orange for O2, green for O3
c1, c2, c3 = "#2563eb", "#ea580c", "#16a34a"

ax.plot(iters, outer1, "o-", color=c1, markersize=4, linewidth=1.5,
        label="Outer 1 (hierarchical, 5 agents)", zorder=3)
ax.plot(iters, outer2, "s-", color=c2, markersize=4, linewidth=1.5,
        label="Outer 2 (ensemble, 6 agents)", zorder=3)
ax.plot(iters, outer3, "D-", color=c3, markersize=4, linewidth=1.5,
        label="Outer 3 (ensemble, 6 agents)", zorder=3)

# Highlight peak at O3/I4 = 70%
ax.annotate("70% (peak)", xy=(4, 0.70), xytext=(5.5, 0.73),
            fontsize=7, fontweight="bold", color=c3,
            arrowprops=dict(arrowstyle="->", color=c3, lw=0.8))

# Mark consolidation at I6 with vertical band
ax.axvspan(5.7, 6.3, alpha=0.10, color="gray", zorder=0)
ax.text(6, 0.32, "consol.", fontsize=6, ha="center", color="gray", style="italic")

# GPT-4o published baseline
ax.axhline(y=0.5896, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
ax.text(8.05, 0.5896, "GPT-4o\nbaseline\n(58.96%)", fontsize=5.5, va="center",
        color="black", alpha=0.6)

ax.set_xlabel("Inner Iteration", fontsize=9)
ax.set_ylabel("Pass Rate", fontsize=9)
ax.set_xticks(iters)
ax.set_ylim(0.28, 0.78)
ax.set_xlim(0.5, 8.8)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax.tick_params(labelsize=7.5)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("figures/fig2_pass_rate_trajectory.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figures/fig2_pass_rate_trajectory.png", bbox_inches="tight", dpi=300)
print("Saved figures/fig2_pass_rate_trajectory.pdf")
