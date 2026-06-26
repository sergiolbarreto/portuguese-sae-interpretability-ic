"""
Análise de robustez do LSI — para rodar no Google Colab após carregar os checkpoints.

Gera:
  1. Tabela de sensibilidade do threshold LSI (0.1 a 0.6)
  2. LSI normalizado por frequência de tokens (controle lexical)
  3. Comparação de ranking: LSI original vs LSI normalizado

Uso no Colab:
  - Certifique-se de que stats_wiki_pt.pt, stats_wiki_en.pt,
    stats_mc4_pt.pt, stats_c4_en.pt estão disponíveis.
  - Ajuste CHECKPOINT_DIR se necessário.
"""

import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração — ajuste o caminho dos checkpoints
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "./checkpoints/"  # ajuste conforme necessário
MIN_ACTS = 100
N_FEATURES = 16_384

# ---------------------------------------------------------------------------
# Carregar dados
# ---------------------------------------------------------------------------
print("Carregando checkpoints...")
stats_wiki_pt = torch.load(CHECKPOINT_DIR + "stats_wiki_pt.pt", weights_only=False)
stats_wiki_en = torch.load(CHECKPOINT_DIR + "stats_wiki_en.pt", weights_only=False)
stats_mc4_pt  = torch.load(CHECKPOINT_DIR + "stats_mc4_pt.pt",  weights_only=False)
stats_c4_en   = torch.load(CHECKPOINT_DIR + "stats_c4_en.pt",   weights_only=False)
print("OK\n")

# Frequências de ativação por feature (mesma lógica do notebook)
freq_wiki_pt = stats_wiki_pt["counts"] / stats_wiki_pt["total_tokens"]
freq_wiki_en = stats_wiki_en["counts"] / stats_wiki_en["total_tokens"]
freq_mc4_pt  = stats_mc4_pt["counts"]  / stats_mc4_pt["total_tokens"]
freq_c4_en   = stats_c4_en["counts"]   / stats_c4_en["total_tokens"]

total_counts = (stats_wiki_pt["counts"] + stats_wiki_en["counts"]
                + stats_mc4_pt["counts"] + stats_c4_en["counts"])
active = total_counts >= MIN_ACTS

# LSI original
lsi_wiki = (freq_wiki_pt - freq_wiki_en) / (freq_wiki_pt + freq_wiki_en + 1e-10)
lsi_web  = (freq_mc4_pt  - freq_c4_en)   / (freq_mc4_pt  + freq_c4_en  + 1e-10)
lsi_combined = (lsi_wiki + lsi_web) / 2

# =========================================================================
# PARTE 1: Análise de sensibilidade do threshold
# =========================================================================
print("=" * 70)
print("PARTE 1: ANÁLISE DE SENSIBILIDADE DO THRESHOLD LSI")
print("=" * 70)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

header = f"{'Threshold':>10} | {'PT wiki':>8} | {'PT web':>8} | {'PT ambos':>9} | {'EN ambos':>9} | {'Cross ambos':>12}"
print(header)
print("-" * len(header))

sensitivity_rows = []
for t in thresholds:
    pt_w = ((lsi_wiki > t) & active).sum().item()
    pt_wc = ((lsi_web > t) & active).sum().item()
    pt_both = ((lsi_wiki > t) & (lsi_web > t) & active).sum().item()
    en_both = ((lsi_wiki < -t) & (lsi_web < -t) & active).sum().item()
    cross_both = ((lsi_wiki.abs() <= t) & (lsi_web.abs() <= t) & active).sum().item()
    sensitivity_rows.append((t, pt_w, pt_wc, pt_both, en_both, cross_both))
    print(f"{t:>10.1f} | {pt_w:>8,} | {pt_wc:>8,} | {pt_both:>9,} | {en_both:>9,} | {cross_both:>12,}")

# Verificar estabilidade do top-50
print("\nEstabilidade do top-50 PT-específicas por threshold:")
lsi_pt_rank = lsi_combined.clone()
lsi_pt_rank[~active] = -2
_, top50_idx = lsi_pt_rank.topk(50)
top50_set = set(top50_idx.tolist())

for t in thresholds:
    pt_at_threshold = set(torch.where((lsi_wiki > t) & (lsi_web > t) & active)[0].tolist())
    overlap = len(top50_set & pt_at_threshold)
    min_lsi = min(lsi_combined[idx].item() for idx in top50_set)
    print(f"  Threshold {t:.1f}: {overlap}/50 do top-50 classificados como PT-específicas "
          f"(min LSI combinado no top-50: {min_lsi:+.4f})")

# =========================================================================
# PARTE 2: LSI normalizado por frequência de tokens
# =========================================================================
print("\n" + "=" * 70)
print("PARTE 2: LSI NORMALIZADO POR FREQUÊNCIA DE TOKENS")
print("=" * 70)
print()
print("Lógica: se uma feature ativa N vezes em M tokens, a frequência bruta")
print("é f = N/M. Se o token associado a essa feature aparece T vezes no")
print("corpus, a frequência normalizada é f_norm = N/T.")
print("Isso controla para tokens que são simplesmente mais frequentes em PT.")
print()
print("Como não temos a contagem exata de tokens individuais por corpus,")
print("usamos uma proxy: a frequência total de ativação de cada feature")
print("no corpus combinado (PT+EN) como denominador de normalização.")
print("Features que ativam muito em ambas as línguas (alta frequência base)")
print("terão seu LSI normalizado atenuado.")
print()

# Proxy de normalização: frequência base = média PT+EN
freq_base_wiki = (freq_wiki_pt + freq_wiki_en) / 2 + 1e-10
freq_base_web  = (freq_mc4_pt  + freq_c4_en)   / 2 + 1e-10

# LSI normalizado: ponderar pela especificidade relativa à frequência base
# Em vez de f_PT - f_EN, usamos (f_PT - f_EN) / freq_base
# Isso penaliza features de alta frequência em ambas as línguas
norm_diff_wiki = (freq_wiki_pt - freq_wiki_en) / freq_base_wiki
norm_diff_web  = (freq_mc4_pt  - freq_c4_en)   / freq_base_web

# Recalcular LSI normalizado mantendo a mesma escala [-1, +1]
# Usar a mesma fórmula mas com frequências normalizadas pela base
# Alternativa mais simples: usar o LSI original mas ranquear por |LSI| * (1 / log(freq_base))
# Vamos usar a abordagem direta: LSI ponderado = LSI * (1 - freq_overlap)
# onde freq_overlap mede quanto a feature é compartilhada

# Abordagem: Jaccard-like normalization
# Se f_PT >> f_EN, o feature é PT-específico independente da frequência
# Se f_PT ≈ f_EN e ambos altos, é cross-lingual de alta frequência
# O LSI original já captura isso! O problema é quando f_PT > f_EN
# simplesmente porque o token subjacente é mais frequente em PT

# Melhor proxy sem dados de unigram: comparar com frequência esperada
# Se feature i ativa em fração p dos tokens PT e fração q dos tokens EN,
# e o corpus PT tem a mesma quantidade de tokens que EN,
# então LSI = (p-q)/(p+q). Isso já é normalizado por tamanho de corpus.
# O confound real seria se o TOKEN que a feature detecta é mais frequente em PT.

# Sem acesso direto à frequência de tokens, fazemos o seguinte:
# Compute um "LSI residual" que desconta a ativação média
# Features com alta ativação em ambas as línguas são penalizadas

# Método: Z-score do LSI condicionado à frequência total
total_freq_wiki = freq_wiki_pt + freq_wiki_en
total_freq_web  = freq_mc4_pt  + freq_c4_en

# Binning por frequência total para calcular LSI esperado por bin
n_bins = 20
active_mask = active.numpy()

def compute_residual_lsi(lsi_tensor, total_freq_tensor, active_mask, n_bins=20):
    """Calcula LSI residual: LSI observado - LSI esperado dado o bin de frequência."""
    lsi_np = lsi_tensor.numpy()
    freq_np = total_freq_tensor.numpy()

    lsi_residual = np.full_like(lsi_np, np.nan)
    active_idx = np.where(active_mask)[0]

    freq_active = freq_np[active_idx]
    lsi_active = lsi_np[active_idx]

    # Log-scale bins (frequências variam em ordens de magnitude)
    log_freq = np.log10(freq_active + 1e-10)
    bin_edges = np.linspace(log_freq.min(), log_freq.max(), n_bins + 1)

    for b in range(n_bins):
        mask = (log_freq >= bin_edges[b]) & (log_freq < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (log_freq >= bin_edges[b]) & (log_freq <= bin_edges[b + 1])
        if mask.sum() < 5:
            continue
        bin_mean = lsi_active[mask].mean()
        bin_std = lsi_active[mask].std()
        if bin_std < 1e-6:
            bin_std = 1.0
        lsi_residual[active_idx[mask]] = (lsi_active[mask] - bin_mean) / bin_std

    return torch.from_numpy(lsi_residual)

lsi_residual_wiki = compute_residual_lsi(lsi_wiki, total_freq_wiki, active_mask)
lsi_residual_web  = compute_residual_lsi(lsi_web,  total_freq_web,  active_mask)
lsi_residual_combined = (lsi_residual_wiki + lsi_residual_web) / 2

# Ranking: comparar top-50 original vs top-50 residual
lsi_res_rank = lsi_residual_combined.clone()
lsi_res_rank[lsi_res_rank.isnan()] = -999
_, top50_residual_idx = lsi_res_rank.topk(50)
top50_residual_set = set(top50_residual_idx.tolist())

overlap = len(top50_set & top50_residual_set)
print(f"Top-50 PT (LSI original) ∩ Top-50 PT (LSI residual): {overlap}/50")
print()

# Tabela comparativa
print(f"{'Rank':>5} | {'ID (orig)':>10} | {'LSI orig':>10} | {'ID (resid)':>10} | {'LSI resid':>10} | {'Match':>6}")
print("-" * 65)
for i in range(50):
    orig_idx = top50_idx[i].item()
    res_idx  = top50_residual_idx[i].item()
    orig_lsi = lsi_combined[orig_idx].item()
    res_lsi  = lsi_residual_combined[res_idx].item()
    match = "✓" if orig_idx == res_idx else ("~" if orig_idx in top50_residual_set else "✗")
    print(f"{i+1:>5} | {orig_idx:>10} | {orig_lsi:>+10.4f} | {res_idx:>10} | {res_lsi:>+10.4f} | {match:>6}")

# Features no top-50 original que SAEM do top-50 residual
dropped = top50_set - top50_residual_set
if dropped:
    print(f"\nFeatures no top-50 original que saem do top-50 residual ({len(dropped)}):")
    for idx in sorted(dropped):
        orig_rank = top50_idx.tolist().index(idx) + 1
        res_val = lsi_residual_combined[idx].item()
        orig_val = lsi_combined[idx].item()
        freq_pt_avg = (freq_wiki_pt[idx].item() + freq_mc4_pt[idx].item()) / 2
        freq_en_avg = (freq_wiki_en[idx].item() + freq_c4_en[idx].item()) / 2
        print(f"  Feature {idx:>5} | rank orig: {orig_rank:>3} | "
              f"LSI orig: {orig_val:+.4f} | LSI resid: {res_val:+.4f} | "
              f"freq PT: {freq_pt_avg:.6f} | freq EN: {freq_en_avg:.6f}")
else:
    print("\nTodas as 50 features do ranking original permanecem no top-50 residual!")

# Features que ENTRAM no top-50 residual (não estavam no original)
gained = top50_residual_set - top50_set
if gained:
    print(f"\nFeatures novas no top-50 residual ({len(gained)}):")
    for idx in sorted(gained):
        res_rank = top50_residual_idx.tolist().index(idx) + 1
        res_val = lsi_residual_combined[idx].item()
        orig_val = lsi_combined[idx].item()
        freq_pt_avg = (freq_wiki_pt[idx].item() + freq_mc4_pt[idx].item()) / 2
        freq_en_avg = (freq_wiki_en[idx].item() + freq_c4_en[idx].item()) / 2
        print(f"  Feature {idx:>5} | rank resid: {res_rank:>3} | "
              f"LSI orig: {orig_val:+.4f} | LSI resid: {res_val:+.4f} | "
              f"freq PT: {freq_pt_avg:.6f} | freq EN: {freq_en_avg:.6f}")

# Correlação Spearman entre rankings
from scipy.stats import spearmanr

# Ranking de todas as features ativas
active_indices = torch.where(active)[0]
lsi_orig_active = lsi_combined[active_indices].numpy()
lsi_res_active = lsi_residual_combined[active_indices].numpy()
valid = ~np.isnan(lsi_res_active)

rho, pval = spearmanr(lsi_orig_active[valid], lsi_res_active[valid])
print(f"\nCorrelação Spearman (ranking LSI orig vs resid, features ativas): ρ = {rho:.4f} (p = {pval:.2e})")

# =========================================================================
# PARTE 3: Resumo para o paper
# =========================================================================
print("\n" + "=" * 70)
print("PARTE 3: TEXTO SUGERIDO PARA O PAPER")
print("=" * 70)
print()
print("--- Para a seção Methodology (LSI threshold justification) ---")
print()
print(f"Sensitivity analysis: varying the threshold from 0.1 to 0.6")
for t, pt_w, pt_wc, pt_both, en_both, cross_both in sensitivity_rows:
    print(f"  |LSI| > {t}: {pt_both:,} PT-specific at both levels")
print()
print(f"Top-50 stability: all 50 selected features have LSI > "
      f"{min(lsi_combined[idx].item() for idx in top50_set):+.4f}")
print()
print("--- Para a seção Discussion (lexical frequency confound) ---")
print()
print(f"Ranking overlap (top-50 original vs frequency-controlled): {overlap}/50")
print(f"Spearman correlation: ρ = {rho:.4f}")
if overlap >= 40:
    print("→ O ranking é robusto à normalização por frequência.")
    print("  A maioria das features PT-específicas não são artefatos de frequência lexical.")
elif overlap >= 30:
    print("→ O ranking é moderadamente robusto. Algumas features podem refletir frequência.")
else:
    print("→ ATENÇÃO: ranking muda substancialmente. Algumas features podem ser artefatos.")
