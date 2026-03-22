"""Generate Figure 1: ABSTRAL Three-Layer Architecture Diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Colors
c_l1 = "#dbeafe"   # light blue (inner loop)
c_l2 = "#fef3c7"   # light yellow (convergence)
c_l3 = "#dcfce7"   # light green (outer loop)
c_border1 = "#2563eb"
c_border2 = "#d97706"
c_border3 = "#16a34a"
c_box = "#f8fafc"
c_text = "#1e293b"
c_arrow = "#475569"

# --- Layer 3: Outer Loop (outermost box) ---
outer = FancyBboxPatch((0.3, 0.3), 9.4, 5.4, boxstyle="round,pad=0.15",
                        facecolor=c_l3, edgecolor=c_border3, linewidth=1.5, alpha=0.35)
ax.add_patch(outer)
ax.text(0.7, 5.35, "Layer 3: Outer Diversity Loop", fontsize=7.5,
        fontweight="bold", color=c_border3, va="top")

# --- Layer 2: Convergence (middle box) ---
mid = FancyBboxPatch((0.7, 0.6), 8.6, 4.3, boxstyle="round,pad=0.12",
                      facecolor=c_l2, edgecolor=c_border2, linewidth=1.3, alpha=0.4)
ax.add_patch(mid)
ax.text(1.1, 4.6, "Layer 2: Convergence Detector", fontsize=7,
        fontweight="bold", color=c_border2, va="top")

# --- Layer 1: Inner Loop (innermost box) ---
inner = FancyBboxPatch((1.1, 0.9), 7.8, 3.3, boxstyle="round,pad=0.1",
                        facecolor=c_l1, edgecolor=c_border1, linewidth=1.3, alpha=0.35)
ax.add_patch(inner)
ax.text(1.5, 3.95, "Layer 1: Inner Loop", fontsize=7,
        fontweight="bold", color=c_border1, va="top")

# --- Phase boxes ---
def phase_box(x, y, w, h, label, sublabel=None, color="#e2e8f0"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor="#64748b", linewidth=0.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + (0.08 if sublabel else 0), label, fontsize=7,
            fontweight="bold", ha="center", va="center", color=c_text)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, fontsize=5,
                ha="center", va="center", color="#64748b", style="italic")

# SKILL.md (left)
phase_box(1.4, 2.0, 1.2, 0.9, "SKILL.md", "$\\mathcal{A}_t$", "#e0f2fe")

# BUILD
phase_box(3.0, 2.0, 1.0, 0.9, "BUILD", "AgentSpec", "#dbeafe")

# RUN
phase_box(4.4, 2.0, 1.0, 0.9, "RUN", "Traces", "#bfdbfe")

# ANALYZE
phase_box(5.8, 2.0, 1.1, 0.9, "ANALYZE", "EC1-5", "#93c5fd")

# UPDATE
phase_box(7.3, 2.0, 1.1, 0.9, "UPDATE", "$\\mathcal{A}_{t+1}$", "#dbeafe")

# --- Arrows between phases ---
arrow_kw = dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                color=c_arrow, lw=1.2, mutation_scale=12)

for (x1, x2) in [(2.6, 3.0), (4.0, 4.4), (5.4, 5.8), (6.9, 7.3)]:
    ax.annotate("", xy=(x2, 2.45), xytext=(x1, 2.45),
                arrowprops=arrow_kw)

# Feedback arrow (UPDATE -> SKILL.md, curved underneath)
ax.annotate("", xy=(2.0, 1.95), xytext=(7.85, 1.95),
            arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                           color=c_border1, lw=1.0, connectionstyle="arc3,rad=-0.35",
                           mutation_scale=12))
ax.text(5.0, 1.15, "iterate", fontsize=5.5, ha="center", color=c_border1,
        style="italic")

# --- Convergence signals ---
ax.text(5.0, 4.35, "C1: skill diff  |  C2: plateau  |  C3: EC exhaust  |  C4: bloat",
        fontsize=5, ha="center", color=c_border2)
ax.annotate("", xy=(5.0, 4.15), xytext=(5.0, 3.3),
            arrowprops=dict(arrowstyle="->,head_width=0.12", color=c_border2,
                           lw=0.8, mutation_scale=10, linestyle="--"))

# --- Outer loop annotation ---
ax.text(5.0, 5.1, "GED repulsion  \u2192  seed new topology  \u2192  clear T, preserve K",
        fontsize=5.5, ha="center", color=c_border3)
ax.annotate("", xy=(5.0, 4.9), xytext=(5.0, 4.5),
            arrowprops=dict(arrowstyle="->,head_width=0.12", color=c_border3,
                           lw=0.8, mutation_scale=10, linestyle="--"))

# --- Output label ---
ax.text(9.3, 0.55, "Output:\nLandscape +\nSKILL.md", fontsize=5, ha="right",
        va="bottom", color="#64748b", style="italic")

fig.tight_layout(pad=0.2)
fig.savefig("figures/fig1_architecture.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figures/fig1_architecture.png", bbox_inches="tight", dpi=300)
print("Saved figures/fig1_architecture.pdf")
