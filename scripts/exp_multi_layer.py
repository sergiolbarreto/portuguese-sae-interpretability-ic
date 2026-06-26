"""
Análise multi-layer — testa se features PT-específicas existem em outras layers.

Roda no Google Colab com T4. Para cada layer testada:
  1. Carrega SAE correspondente do Gemma Scope
  2. Processa 1M tokens por corpus (4M total) para viabilidade
  3. Computa LSI e conta features PT-específicas
  4. Compara ranking com layer 13

Uso:
  1. Faça upload deste script para o Colab
  2. Execute: !python exp_multi_layer.py
  3. Resultados salvos em exp_multi_layer_results.json

Tempo estimado: ~3h no T4 (5 layers × ~35 min cada)
Requer: pip install transformer_lens sae_lens torch numpy datasets scipy tqdm
"""

import torch
import numpy as np
import json
import time
import os
from tqdm import tqdm
from scipy.stats import spearmanr

SAVE_DIR = "./"
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# CONFIG
# =========================================================================
LAYERS_TO_TEST = [5, 9, 13, 17, 21]
N_TOKENS = 1_000_000  # 1M per corpus (vs 5M in full analysis) for speed
SEQ_LEN = 128   # reduced to fit T4 VRAM
BATCH_SIZE = 4  # reduced to fit T4 VRAM
MIN_ACTS = 20  # lower threshold since we have 1M instead of 5M tokens
LSI_THRESHOLD = 0.3

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"

# =========================================================================
# LOAD MODEL (once, shared across all layers)
# =========================================================================
print("=" * 70)
print("SETUP: Carregando modelo")
print("=" * 70)

from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset

print("Carregando Gemma 2 2B...")
model = HookedSAETransformer.from_pretrained_no_processing(
    "gemma-2-2b",
    device=device,
    dtype=torch.float16,
)
print(f"Modelo: {model.cfg.model_name} | Layers: {model.cfg.n_layers} | d_model: {model.cfg.d_model}")

tokenizer = model.tokenizer

# =========================================================================
# COLLECT TOKENS (once, shared across all layers)
# =========================================================================
print("\n" + "=" * 70)
print("COLETANDO TOKENS")
print("=" * 70)

def collect_tokens(dataset, n_tokens, text_field="text", desc="Tokenizando"):
    all_ids = []
    n_articles = 0
    for article in tqdm(dataset, desc=desc):
        text = article[text_field]
        if not text or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        n_articles += 1
        if len(all_ids) >= n_tokens:
            break
    all_ids = all_ids[:n_tokens]
    n_seqs = len(all_ids) // SEQ_LEN
    tokens = torch.tensor(all_ids[:n_seqs * SEQ_LEN], dtype=torch.long).reshape(n_seqs, SEQ_LEN)
    print(f"  {n_articles} textos -> {tokens.numel():,} tokens ({tokens.shape[0]} seqs)")
    return tokens

print("\nWikipedia PT...")
wiki_pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
tokens_wiki_pt = collect_tokens(wiki_pt, N_TOKENS, desc="Wiki PT")

print("Wikipedia EN...")
wiki_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
tokens_wiki_en = collect_tokens(wiki_en, N_TOKENS, desc="Wiki EN")

print("FineWeb-2 PT...")
web_pt = load_dataset("HuggingFaceFW/fineweb-2", "por_Latn", split="train", streaming=True)
tokens_mc4_pt = collect_tokens(web_pt, N_TOKENS, desc="FineWeb PT")

print("FineWeb EN...")
web_en = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
tokens_c4_en = collect_tokens(web_en, N_TOKENS, desc="FineWeb EN")

total_tokens = (tokens_wiki_pt.numel() + tokens_wiki_en.numel()
                + tokens_mc4_pt.numel() + tokens_c4_en.numel())
print(f"\nTotal: {total_tokens:,} tokens ({N_TOKENS:,} per corpus)")


# =========================================================================
# COMPUTE STATS PER LAYER
# =========================================================================

def compute_feature_stats(model, sae, tokens, hook_name, batch_size=BATCH_SIZE, desc=""):
    n_features = sae.cfg.d_sae
    counts = torch.zeros(n_features, device=device)
    sums = torch.zeros(n_features, device=device)
    maxvals = torch.zeros(n_features, device=device)
    total = 0

    n_batches = (len(tokens) + batch_size - 1) // batch_size
    t0 = time.time()
    for i in tqdm(range(0, len(tokens), batch_size), desc=desc, total=n_batches):
        batch = tokens[i:i+batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]
        feat_acts = sae.encode(acts)

        active = feat_acts > 0
        counts += active.float().sum(dim=(0, 1))
        sums += feat_acts.sum(dim=(0, 1))
        maxvals = torch.max(maxvals, feat_acts.amax(dim=(0, 1)))
        total += batch.numel()

        del cache, acts, feat_acts, active
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  {total:,} tokens | {(counts > 0).sum().item():,} features ativas | {elapsed:.0f}s")
    return {"counts": counts.cpu(), "sums": sums.cpu(), "max": maxvals.cpu(), "total_tokens": total}


all_layer_results = {}

for layer in LAYERS_TO_TEST:
    print("\n" + "=" * 70)
    print(f"LAYER {layer}")
    print("=" * 70)

    sae_id = f"layer_{layer}/width_16k/canonical"
    hook_name = f"blocks.{layer}.hook_resid_post"

    print(f"Carregando SAE: {sae_id}...")
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )
    print(f"SAE: {sae.cfg.d_sae} features | hook: {hook_name}")

    t0_layer = time.time()

    stats_wiki_pt = compute_feature_stats(model, sae, tokens_wiki_pt, hook_name, desc=f"L{layer} Wiki PT")
    stats_wiki_en = compute_feature_stats(model, sae, tokens_wiki_en, hook_name, desc=f"L{layer} Wiki EN")
    stats_mc4_pt  = compute_feature_stats(model, sae, tokens_mc4_pt,  hook_name, desc=f"L{layer} FineWeb PT")
    stats_c4_en   = compute_feature_stats(model, sae, tokens_c4_en,   hook_name, desc=f"L{layer} FineWeb EN")

    # Compute LSI
    freq_wiki_pt = stats_wiki_pt["counts"] / stats_wiki_pt["total_tokens"]
    freq_wiki_en = stats_wiki_en["counts"] / stats_wiki_en["total_tokens"]
    freq_mc4_pt  = stats_mc4_pt["counts"]  / stats_mc4_pt["total_tokens"]
    freq_c4_en   = stats_c4_en["counts"]   / stats_c4_en["total_tokens"]

    total_counts = (stats_wiki_pt["counts"] + stats_wiki_en["counts"]
                    + stats_mc4_pt["counts"] + stats_c4_en["counts"])
    alive = total_counts > 0
    active = total_counts >= MIN_ACTS

    lsi_wiki = (freq_wiki_pt - freq_wiki_en) / (freq_wiki_pt + freq_wiki_en + 1e-10)
    lsi_web  = (freq_mc4_pt  - freq_c4_en)   / (freq_mc4_pt  + freq_c4_en  + 1e-10)
    lsi_combined = (lsi_wiki + lsi_web) / 2

    # Counts
    pt_wiki = ((lsi_wiki > LSI_THRESHOLD) & active).sum().item()
    pt_web  = ((lsi_web > LSI_THRESHOLD) & active).sum().item()
    pt_both = ((lsi_wiki > LSI_THRESHOLD) & (lsi_web > LSI_THRESHOLD) & active).sum().item()
    en_both = ((lsi_wiki < -LSI_THRESHOLD) & (lsi_web < -LSI_THRESHOLD) & active).sum().item()
    cross_both = ((lsi_wiki.abs() <= LSI_THRESHOLD) & (lsi_web.abs() <= LSI_THRESHOLD) & active).sum().item()

    lsi_active = lsi_combined[active]

    elapsed_layer = time.time() - t0_layer

    layer_data = {
        "layer": layer,
        "n_features": int(sae.cfg.d_sae),
        "alive": int(alive.sum().item()),
        "active": int(active.sum().item()),
        "pt_wiki": pt_wiki,
        "pt_web": pt_web,
        "pt_both": pt_both,
        "en_both": en_both,
        "cross_both": cross_both,
        "mean_lsi": float(lsi_active.mean().item()),
        "median_lsi": float(lsi_active.median().item()),
        "std_lsi": float(lsi_active.std().item()),
        "elapsed_s": float(elapsed_layer),
    }

    # Save LSI tensor for cross-layer correlation
    layer_data["_lsi_combined"] = lsi_combined

    all_layer_results[layer] = layer_data

    print(f"\n  Layer {layer} summary:")
    print(f"    Active features: {layer_data['active']:,}")
    print(f"    PT-specific (both): {pt_both}")
    print(f"    EN-specific (both): {en_both}")
    print(f"    Cross-lingual (both): {cross_both}")
    print(f"    Mean LSI: {layer_data['mean_lsi']:.4f}")
    print(f"    Time: {elapsed_layer:.0f}s")

    # Save checkpoint per layer
    torch.save({
        "stats_wiki_pt": stats_wiki_pt,
        "stats_wiki_en": stats_wiki_en,
        "stats_mc4_pt": stats_mc4_pt,
        "stats_c4_en": stats_c4_en,
        "lsi_wiki": lsi_wiki,
        "lsi_web": lsi_web,
        "lsi_combined": lsi_combined,
        "active": active,
    }, os.path.join(SAVE_DIR, f"stats_layer{layer}.pt"))
    print(f"    Checkpoint: stats_layer{layer}.pt")

    del sae, stats_wiki_pt, stats_wiki_en, stats_mc4_pt, stats_c4_en
    if device == "cuda":
        torch.cuda.empty_cache()


# =========================================================================
# CROSS-LAYER ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("CROSS-LAYER ANALYSIS")
print("=" * 70)

ref_layer = 13
ref_lsi = all_layer_results[ref_layer]["_lsi_combined"]

print(f"\nCorrelação Spearman com layer {ref_layer}:")
correlations = {}

for layer in LAYERS_TO_TEST:
    if layer == ref_layer:
        correlations[layer] = {"rho": 1.0, "pval": 0.0}
        continue

    layer_lsi = all_layer_results[layer]["_lsi_combined"]

    # Both must be active
    both_valid = (ref_lsi.abs() < 2) & (layer_lsi.abs() < 2)
    if both_valid.sum() < 100:
        correlations[layer] = {"rho": float("nan"), "pval": float("nan")}
        continue

    rho, pval = spearmanr(
        ref_lsi[both_valid].numpy(),
        layer_lsi[both_valid].numpy(),
    )
    correlations[layer] = {"rho": float(rho), "pval": float(pval)}
    print(f"  Layer {layer:>2} vs Layer {ref_layer}: ρ = {rho:.4f} (p = {pval:.2e})")

# Summary table
print("\n" + "-" * 80)
print(f"{'Layer':>6} | {'Active':>8} | {'PT both':>8} | {'EN both':>8} | {'Cross':>8} | {'Mean LSI':>10} | {'ρ vs L13':>10}")
print("-" * 80)
for layer in LAYERS_TO_TEST:
    d = all_layer_results[layer]
    rho = correlations[layer]["rho"]
    rho_str = f"{rho:.4f}" if not np.isnan(rho) else "N/A"
    print(f"{layer:>6} | {d['active']:>8,} | {d['pt_both']:>8} | {d['en_both']:>8} | {d['cross_both']:>8} | {d['mean_lsi']:>+10.4f} | {rho_str:>10}")

# =========================================================================
# SAVE RESULTS
# =========================================================================
print("\n" + "=" * 70)
print("SALVANDO RESULTADOS")
print("=" * 70)

# Remove non-serializable tensors
export_results = {}
for layer, data in all_layer_results.items():
    export_results[str(layer)] = {k: v for k, v in data.items() if not k.startswith("_")}

export_results["correlations_vs_layer13"] = {str(k): v for k, v in correlations.items()}
export_results["config"] = {
    "layers_tested": LAYERS_TO_TEST,
    "n_tokens_per_corpus": N_TOKENS,
    "min_acts": MIN_ACTS,
    "lsi_threshold": LSI_THRESHOLD,
    "sae_release": SAE_RELEASE,
    "ref_layer": ref_layer,
}

output_path = os.path.join(SAVE_DIR, "exp_multi_layer_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(export_results, f, ensure_ascii=False, indent=2)

print(f"Resultados salvos em: {output_path}")
print(f"Checkpoints por layer: stats_layer{{5,9,13,17,21}}.pt")
print("\nPronto!")
