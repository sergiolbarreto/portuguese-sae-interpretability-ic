"""Generate Figure 7: Multi-layer U-shaped pattern visualization.

Writes to ../figures/ by default (override with env var FIG_OUT_DIR).
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

OUT_DIR = os.environ.get("FIG_OUT_DIR", "../figures")
os.makedirs(OUT_DIR, exist_ok=True)

layers = [5, 9, 13, 17, 21]
pt_counts = [1415, 1224, 651, 768, 1058]
mean_lsi = [-0.031, -0.171, -0.322, -0.355, -0.320]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

ax1.plot(layers, pt_counts, 'o-', color='#c0392b', linewidth=2, markersize=8)
ax1.axvline(x=13, color='gray', linestyle='--', alpha=0.5, label='Layer 13')
ax1.set_xlabel('Layer')
ax1.set_ylabel('PT-specific features (LSI > 0.3)')
ax1.set_title('(A) Feature count by layer')
ax1.set_xticks(layers)
ax1.legend()
ax1.grid(True, alpha=0.3)
for i, (x, y) in enumerate(zip(layers, pt_counts)):
    ax1.annotate(str(y), (x, y), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=10)

ax2.plot(layers, mean_lsi, 's-', color='#2980b9', linewidth=2, markersize=8)
ax2.axvline(x=13, color='gray', linestyle='--', alpha=0.5, label='Layer 13')
ax2.axhline(y=0, color='black', linestyle=':', alpha=0.3)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Mean LSI')
ax2.set_title('(B) Mean LSI by layer')
ax2.set_xticks(layers)
ax2.legend()
ax2.grid(True, alpha=0.3)
for i, (x, y) in enumerate(zip(layers, mean_lsi)):
    ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0, -18 if y < -0.2 else 12), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_multilayer_pattern.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'fig_multilayer_pattern.pdf'),
            bbox_inches='tight')
print(f"Saved: {OUT_DIR}/fig_multilayer_pattern.png / .pdf")
