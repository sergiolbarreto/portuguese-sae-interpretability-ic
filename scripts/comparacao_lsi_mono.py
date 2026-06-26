"""
Comparação LSI vs Monolingualidade (Deng et al., ACL 2025).

Computa a métrica de monolingualidade baseada em entropia sobre a massa
de ativação e compara com o LSI (baseado em frequência de ativação).

Roda localmente — só precisa dos checkpoints .pt.
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

CHECKPOINT_DIR = "./checkpoints/"
MIN_ACTS = 100

print("Carregando checkpoints...")
stats_wiki_pt = torch.load(CHECKPOINT_DIR + "stats_wiki_pt.pt", weights_only=False)
stats_wiki_en = torch.load(CHECKPOINT_DIR + "stats_wiki_en.pt", weights_only=False)
stats_mc4_pt  = torch.load(CHECKPOINT_DIR + "stats_mc4_pt.pt",  weights_only=False)
stats_c4_en   = torch.load(CHECKPOINT_DIR + "stats_c4_en.pt",   weights_only=False)
print("OK\n")

# =========================================================================
# 1. LSI original (frequência de ativação)
# =========================================================================
freq_wiki_pt = stats_wiki_pt["counts"] / stats_wiki_pt["total_tokens"]
freq_wiki_en = stats_wiki_en["counts"] / stats_wiki_en["total_tokens"]
freq_mc4_pt  = stats_mc4_pt["counts"]  / stats_mc4_pt["total_tokens"]
freq_c4_en   = stats_c4_en["counts"]   / stats_c4_en["total_tokens"]

total_counts = (stats_wiki_pt["counts"] + stats_wiki_en["counts"]
                + stats_mc4_pt["counts"] + stats_c4_en["counts"])
active = total_counts >= MIN_ACTS

lsi_wiki = (freq_wiki_pt - freq_wiki_en) / (freq_wiki_pt + freq_wiki_en + 1e-10)
lsi_web  = (freq_mc4_pt  - freq_c4_en)   / (freq_mc4_pt  + freq_c4_en  + 1e-10)
lsi_combined = (lsi_wiki + lsi_web) / 2

# =========================================================================
# 2. Monolingualidade (Deng et al.) — baseada em massa de ativação
# =========================================================================
# Soma total de ativações por feature por corpus
sums_wiki_pt = stats_wiki_pt["sums"]
sums_wiki_en = stats_wiki_en["sums"]
sums_mc4_pt  = stats_mc4_pt["sums"]
sums_c4_en   = stats_c4_en["sums"]

# Agrupar por língua: PT = wiki_pt + mc4_pt, EN = wiki_en + c4_en
sum_pt = sums_wiki_pt + sums_mc4_pt
sum_en = sums_wiki_en + sums_c4_en
sum_total = sum_pt + sum_en + 1e-10

# Fração da massa de ativação por língua
p_pt = sum_pt / sum_total
p_en = sum_en / sum_total

# Entropia de Shannon (base 2, para L=2 línguas → max = 1 bit)
eps = 1e-10
entropy = -(p_pt * torch.log2(p_pt + eps) + p_en * torch.log2(p_en + eps))
max_entropy = 1.0  # log2(2) = 1

# Monolingualidade = 1 - H/H_max (0 = cross-lingual, 1 = monolingual)
monolinguality = 1.0 - entropy / max_entropy

# Versão direcional: positiva se PT-dominant, negativa se EN-dominant
mono_directed = monolinguality * torch.sign(sum_pt - sum_en)

# Também computar um "LSI por magnitude" (análogo direto do LSI mas com sums)
lsi_magnitude = (sum_pt - sum_en) / (sum_pt + sum_en + 1e-10)

# =========================================================================
# 3. Comparações
# =========================================================================
mask = active.numpy()
lsi_vals = lsi_combined.numpy()[mask]
mono_vals = monolinguality.numpy()[mask]
mono_dir_vals = mono_directed.numpy()[mask]
lsi_mag_vals = lsi_magnitude.numpy()[mask]

n_active = mask.sum()
print(f"Features ativas (≥{MIN_ACTS} ativações): {n_active:,}")

# Spearman correlations
rho_mono, p_mono = spearmanr(lsi_vals, mono_dir_vals)
rho_mag, p_mag = spearmanr(lsi_vals, lsi_mag_vals)

print(f"\n{'='*60}")
print("CORRELAÇÕES (Spearman)")
print(f"{'='*60}")
print(f"  LSI vs Monolingualidade (direcional): ρ = {rho_mono:.4f} (p = {p_mono:.2e})")
print(f"  LSI vs LSI-magnitude (sums):          ρ = {rho_mag:.4f}  (p = {p_mag:.2e})")

# =========================================================================
# 4. Classificação: quais features mudam de categoria?
# =========================================================================
LSI_THRESH = 0.3
MONO_THRESH = 0.3  # mesma escala para comparação justa

lsi_pt = lsi_vals > LSI_THRESH
mono_pt = mono_dir_vals > MONO_THRESH
lsi_en = lsi_vals < -LSI_THRESH
mono_en = mono_dir_vals < -MONO_THRESH

agree_pt = (lsi_pt & mono_pt).sum()
only_lsi_pt = (lsi_pt & ~mono_pt).sum()
only_mono_pt = (~lsi_pt & mono_pt).sum()

agree_en = (lsi_en & mono_en).sum()
only_lsi_en = (lsi_en & ~mono_en).sum()
only_mono_en = (~lsi_en & mono_en).sum()

print(f"\n{'='*60}")
print("CONCORDÂNCIA DE CLASSIFICAÇÃO (threshold = 0.3)")
print(f"{'='*60}")
print(f"  PT-specific: ambas={agree_pt}, só LSI={only_lsi_pt}, só Mono={only_mono_pt}")
print(f"  EN-specific: ambas={agree_en}, só LSI={only_lsi_en}, só Mono={only_mono_en}")

# =========================================================================
# 5. Features validadas por probes — consistência
# =========================================================================
PROBE_FEATURES = {
    "gender (1215)": 1215,
    "crase (4584)": 4584,
    "enclisis (2817)": 2817,
    "proclisis (6215)": 6215,
    "contraction_do (2294)": 2294,
    "clitic_lhe (15135)": 15135,
    "preposition_por (10478)": 10478,
    "personal_inf (10349)": 10349,
}

print(f"\n{'='*60}")
print("FEATURES VALIDADAS POR PROBES")
print(f"{'='*60}")
print(f"  {'Feature':<25} {'LSI':>8} {'Mono(dir)':>10} {'LSI-mag':>10} {'Consistente':>12}")
print(f"  {'-'*65}")

all_consistent = True
for name, fid in PROBE_FEATURES.items():
    lsi_v = lsi_combined[fid].item()
    mono_v = mono_directed[fid].item()
    mag_v = lsi_magnitude[fid].item()
    consistent = "✓" if (lsi_v > LSI_THRESH and mono_v > MONO_THRESH) else "✗"
    if consistent == "✗":
        all_consistent = False
    print(f"  {name:<25} {lsi_v:>+8.4f} {mono_v:>+10.4f} {mag_v:>+10.4f} {consistent:>12}")

print(f"\n  Todas consistentes: {'SIM' if all_consistent else 'NÃO'}")

# =========================================================================
# 6. Top-10 features com maior divergência LSI vs Mono
# =========================================================================
divergence = np.abs(lsi_vals - mono_dir_vals)
top_div_idx = np.argsort(divergence)[-10:][::-1]

# Map back to feature IDs
active_indices = np.where(mask)[0]

print(f"\n{'='*60}")
print("TOP-10 FEATURES COM MAIOR DIVERGÊNCIA LSI vs MONO")
print(f"{'='*60}")
print(f"  {'Feature ID':>10} {'LSI':>8} {'Mono(dir)':>10} {'Δ':>8}")
print(f"  {'-'*40}")
for idx in top_div_idx:
    fid = active_indices[idx]
    print(f"  {fid:>10} {lsi_vals[idx]:>+8.4f} {mono_dir_vals[idx]:>+10.4f} {divergence[idx]:>8.4f}")

# =========================================================================
# 7. Resumo para o paper
# =========================================================================
print(f"\n{'='*60}")
print("RESUMO PARA O PAPER")
print(f"{'='*60}")
print(f"  Spearman LSI vs Mono(direcional): ρ = {rho_mono:.4f}")
print(f"  Spearman LSI vs LSI-magnitude:    ρ = {rho_mag:.4f}")
print(f"  PT-specific concordância: {agree_pt}/{agree_pt + only_lsi_pt + only_mono_pt} ({100*agree_pt/(agree_pt + only_lsi_pt + only_mono_pt):.1f}%)")
print(f"  Probe features consistentes: {sum(1 for n, f in PROBE_FEATURES.items() if lsi_combined[f].item() > LSI_THRESH and mono_directed[f].item() > MONO_THRESH)}/{len(PROBE_FEATURES)}")
