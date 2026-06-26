"""
Experimentos de robustez — Layer 13.

Roda no Google Colab com T4. Carrega modelo + SAE uma vez e executa 4 blocos:
  1. Multi-seed gender ablation
  2. Probes expandidos + bootstrap CI 95%
  3. Multi-feature steering
  4. Register steering quantitativo

Uso:
  1. Faça upload deste script para o Colab
  2. Execute: !python exp_robustness_layer13.py
  3. Resultados salvos em exp_robustness_results.json

Requer: pip install transformer_lens sae_lens torch numpy scipy
"""

import torch
import numpy as np
import random
import json
import time
import os

# Global seeds for reproducibility (bootstrap CIs, multi-feature steering
# random feature selection). Multi-seed gender ablation overrides this with
# the loop variable from SEEDS.
GLOBAL_SEED = int(os.environ.get("EXP_SEED", "42"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

SAVE_DIR = os.environ.get("EXP_SAVE_DIR", "./")
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# SETUP: Load model + SAE (once)
# =========================================================================
print("=" * 70)
print("SETUP: Carregando modelo e SAE")
print("=" * 70)

from sae_lens import SAE, HookedSAETransformer

LAYER = 13
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = f"layer_{LAYER}/width_16k/canonical"

print("Carregando Gemma 2 2B...")
model = HookedSAETransformer.from_pretrained_no_processing(
    "gemma-2-2b",
    device=device,
    dtype=torch.float16,
)
print(f"Modelo: {model.cfg.model_name} | Layers: {model.cfg.n_layers}")

print(f"Carregando SAE: {SAE_ID}...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device,
)
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
print(f"SAE: {sae.cfg.d_sae} features | hook: {HOOK_NAME}")

tokenizer = model.tokenizer


# =========================================================================
# UTILITY FUNCTIONS (from original notebooks)
# =========================================================================

def generate_with_steering(model, sae, tokenizer, prompt, feature_ids,
                          multipliers, max_new_tokens=100, temperature=0.7,
                          hook_name=HOOK_NAME):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    feature_ids_t = torch.tensor(feature_ids, device=device)
    multipliers_t = torch.tensor(multipliers, dtype=torch.float16, device=device)

    def steering_hook(value, hook):
        with torch.no_grad():
            sae_input = value
            sae_acts = sae.encode(sae_input)
            modified_acts = sae_acts.clone()
            for fid, mult in zip(feature_ids_t, multipliers_t):
                modified_acts[:, :, fid] = modified_acts[:, :, fid] * mult
            reconstructed = sae.decode(modified_acts)
            error = sae_input - sae.decode(sae_acts)
            return reconstructed + error

    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                generated,
                fwd_hooks=[(hook_name, steering_hook)],
            )
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def generate_baseline(model, tokenizer, prompt, max_new_tokens=100,
                      temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def measure_feature_activations(model, sae, tokenizer, text, feature_ids,
                               hook_name=HOOK_NAME):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    activations = {}

    def capture_hook(value, hook):
        activations["resid"] = value.detach()
        return value

    with torch.no_grad():
        model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, capture_hook)])

    resid = activations["resid"]
    sae_acts = sae.encode(resid)

    results = {}
    for fid in feature_ids:
        acts = sae_acts[0, :, fid].detach().cpu().numpy()
        results[fid] = {
            "mean": float(np.mean(acts)),
            "max": float(np.max(acts)),
            "nonzero_frac": float(np.mean(acts > 0)),
        }
    return results


@torch.no_grad()
def process_text(model, sae, text, hook_name, feature_indices=None):
    tokens = model.tokenizer.encode(text, return_tensors="pt").to(device)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
    acts = cache[hook_name]
    feat_acts = sae.encode(acts)

    max_acts = feat_acts[0].max(dim=0).values.cpu()
    mean_acts = feat_acts[0].mean(dim=0).cpu()

    result = {
        "max_acts_all": max_acts,
        "mean_acts_all": mean_acts,
    }

    if feature_indices is not None:
        fi = torch.tensor(feature_indices) if not isinstance(feature_indices, torch.Tensor) else feature_indices
        result["max_acts"] = max_acts[fi]
        result["mean_acts"] = mean_acts[fi]
        result["token_acts"] = feat_acts[0, :, fi].cpu()

    del cache, acts, feat_acts
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# =========================================================================
# FEATURE IDS
# =========================================================================
#
# Os IDs abaixo foram identificados no pipeline principal (notebooks
# phase3_full_analysis.ipynb + phase4_probes.ipynb), processando 5M tokens
# por corpus (Wikipedia PT/EN, FineWeb-2 PT, FineWeb EN). Cada feature é
# referenciada em uma seção específica do paper:
#
#   1215  → Gender (Tab. 6, §4.7.1 — benchmark de gênero, ablação 20/20)
#   4584  → Crase (Tab. 3 top-features, §4.7.6 amplificação)
#   2817  → Enclisis (Tab. 3 top-features, §4.7.3 probes)
#   6215  → Proclisis (Tab. 3 top-features, único PT-genuíno vs ES)
#   2294  → Contração 'do' (Tab. 3 + §4.7.8 PT-específica vs espanhol)
#   15135 → Clítico 'lhe' (Tab. 3, ES-dominant na §4.7.8)
#   10478 → Preposição 'por' (Tab. 3 + §4.7.5)
#   10349 → Infinitivo pessoal (Tab. 3 top-features)
#
# Para reproduzir a derivação, ver results/exp_spanish_control_results.json
# (probe_features) e a tabela 3 do paper.
#
# IMPORTANTE: estes IDs assumem o SAE canônico Gemma Scope 2B layer 13 16k.
# Re-treinar o SAE OU usar outra release vai produzir IDs diferentes.
FEATURES_PT = {
    "gender": 1215,
    "crase": 4584,
    "enclisis": 2817,
    "proclisis": 6215,
    "contraction_do": 2294,
    "clitic_lhe": 15135,
    "preposition_por": 10478,
    "personal_infinitive": 10349,
}

FEATURES_REGISTER = {
    "colloquial": [1082, 15135],
    "legal": [2294, 12269],
    "scientific": [5880],
    "journalistic": [16057, 10478],
}

ALL_REGISTER_FEATURE_IDS = list(set(
    fid for fids in FEATURES_REGISTER.values() for fid in fids
))

ALL_RESULTS = {}

# =========================================================================
# BLOCO 1: Multi-seed Gender Ablation
# =========================================================================
print("\n" + "=" * 70)
print("BLOCO 1: MULTI-SEED GENDER ABLATION")
print("=" * 70)

SEEDS = [42, 123, 456, 789, 1337]
GENDER_PROMPTS = [
    "A menina bonita foi à",
    "A diretora da empresa apresentou a nova",
    "A professora explicou que a aluna",
    "A gata preta dormia tranquila no",
]

gender_results = []

for seed in SEEDS:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    seed_data = {"seed": seed, "prompts": []}

    for prompt in GENDER_PROMPTS:
        baseline = generate_baseline(model, tokenizer, prompt,
                                     max_new_tokens=50, temperature=0.3)
        ablated = generate_with_steering(model, sae, tokenizer, prompt,
                                         feature_ids=[FEATURES_PT["gender"]],
                                         multipliers=[0.0],
                                         max_new_tokens=50, temperature=0.3)

        seed_data["prompts"].append({
            "prompt": prompt,
            "baseline": baseline,
            "ablated": ablated,
        })
        print(f"  Seed {seed} | {prompt[:30]}...")
        print(f"    Baseline: {baseline[len(prompt):len(prompt)+60]}...")
        print(f"    Ablated:  {ablated[len(prompt):len(prompt)+60]}...")

    gender_results.append(seed_data)

ALL_RESULTS["multi_seed_gender"] = gender_results
print(f"\nBloco 1 concluído: {len(SEEDS)} seeds × {len(GENDER_PROMPTS)} prompts")


# =========================================================================
# BLOCO 2: Expanded Probes + Bootstrap
# =========================================================================
print("\n" + "=" * 70)
print("BLOCO 2: PROBES EXPANDIDOS + BOOTSTRAP")
print("=" * 70)

EXPANDED_PROBES = {
    "genero": {
        "name": "Concordância de Gênero",
        "type": "pairs",
        "pairs": [
            ("O menino bonito brincava no parque.", "A menina bonita brincava no parque."),
            ("O professor alto entrou na sala de aula.", "A professora alta entrou na sala de aula."),
            ("O gato preto dormia tranquilo no sofá.", "A gata preta dormia tranquila no sofá."),
            ("O aluno aplicado foi aprovado no exame.", "A aluna aplicada foi aprovada no exame."),
            ("O médico brasileiro foi premiado ontem.", "A médica brasileira foi premiada ontem."),
            ("O escritor famoso publicou um novo livro.", "A escritora famosa publicou um novo livro."),
            ("O cantor jovem apresentou-se no festival.", "A cantora jovem apresentou-se no festival."),
            ("O diretor responsável assinou o documento.", "A diretora responsável assinou o documento."),
            ("O vizinho simpático cumprimentou todos.", "A vizinha simpática cumprimentou todos."),
            ("O irmão mais velho viajou para o exterior.", "A irmã mais velha viajou para o exterior."),
            ("O engenheiro competente resolveu o problema.", "A engenheira competente resolveu o problema."),
            ("O enfermeiro dedicado cuidou do paciente.", "A enfermeira dedicada cuidou do paciente."),
            ("O juiz rigoroso proferiu a sentença.", "A juíza rigorosa proferiu a sentença."),
            ("O garoto esperto encontrou a solução.", "A garota esperta encontrou a solução."),
            ("O funcionário novo começou hoje.", "A funcionária nova começou hoje."),
            ("O avô carinhoso contou uma história.", "A avó carinhosa contou uma história."),
        ],
        "labels": ("masculino", "feminino"),
    },
    "concordancia": {
        "name": "Concordância Nominal",
        "type": "pairs",
        "pairs": [
            ("As casas grandes foram vendidas rapidamente.", "As casas grande foram vendida rapidamente."),
            ("Os meninos bonitos correram pelo campo.", "Os meninos bonito correram pelo campo."),
            ("As flores vermelhas enfeitavam todo o jardim.", "As flores vermelha enfeitavam todo o jardim."),
            ("Os livros antigos estavam arrumados na estante.", "Os livros antigo estavam arrumados na estante."),
            ("As ruas estreitas dificultavam o trânsito.", "As ruas estreita dificultavam o trânsito."),
            ("Os prédios altos dominavam a paisagem.", "Os prédios alto dominavam a paisagem."),
            ("As portas pesadas foram abertas com esforço.", "As portas pesada foram aberta com esforço."),
            ("Os carros novos brilhavam no estacionamento.", "Os carros novo brilhavam no estacionamento."),
            ("As montanhas nevadas encantavam os turistas.", "As montanhas nevada encantavam os turistas."),
            ("Os caminhos tortuosos levavam ao vilarejo.", "Os caminhos tortuoso levavam ao vilarejo."),
            ("As águas cristalinas refletiam o céu azul.", "As águas cristalina refletiam o céu azul."),
            ("Os quadros antigos decoravam as paredes brancas.", "Os quadros antigo decoravam as paredes branca."),
        ],
        "labels": ("correto", "incorreto"),
    },
    "conjugacao": {
        "name": "Conjugação Verbal",
        "type": "sets",
        "sets": [
            ["Eu corro todos os dias no parque.", "Tu corres todos os dias no parque.",
             "Ele corre todos os dias no parque.", "Nós corremos todos os dias no parque.",
             "Eles correm todos os dias no parque."],
            ["Eu falo português fluentemente.", "Tu falas português fluentemente.",
             "Ele fala português fluentemente.", "Nós falamos português fluentemente.",
             "Eles falam português fluentemente."],
            ["Eu como frutas pela manhã.", "Tu comes frutas pela manhã.",
             "Ele come frutas pela manhã.", "Nós comemos frutas pela manhã.",
             "Eles comem frutas pela manhã."],
            ["Eu escrevo cartas para os amigos.", "Tu escreves cartas para os amigos.",
             "Ele escreve cartas para os amigos.", "Nós escrevemos cartas para os amigos.",
             "Eles escrevem cartas para os amigos."],
            ["Eu durmo cedo durante a semana.", "Tu dormes cedo durante a semana.",
             "Ele dorme cedo durante a semana.", "Nós dormimos cedo durante a semana.",
             "Eles dormem cedo durante a semana."],
            ["Eu leio jornais todas as manhãs.", "Tu lês jornais todas as manhãs.",
             "Ele lê jornais todas as manhãs.", "Nós lemos jornais todas as manhãs.",
             "Eles leem jornais todas as manhãs."],
        ],
        "labels": ["eu", "tu", "ele/ela", "nós", "eles/elas"],
    },
    "crase": {
        "name": "Crase",
        "type": "pairs",
        "pairs": [
            ("Vou à escola todos os dias.", "Vou a escola todos os dias."),
            ("Refiro-me à questão principal do debate.", "Refiro-me a questão principal do debate."),
            ("Ele chegou à cidade ontem à noite.", "Ele chegou a cidade ontem a noite."),
            ("À medida que o tempo passava, tudo mudava.", "A medida que o tempo passava, tudo mudava."),
            ("Dedicou-se à pesquisa científica.", "Dedicou-se a pesquisa científica."),
            ("Fomos à praia no fim de semana.", "Fomos a praia no fim de semana."),
            ("Ele fez referência à obra de Machado.", "Ele fez referência a obra de Machado."),
            ("Graças à chuva, a seca acabou.", "Graças a chuva, a seca acabou."),
            ("À luz dos fatos, a decisão foi justa.", "A luz dos fatos, a decisão foi justa."),
            ("Entregou o relatório à coordenadora.", "Entregou o relatório a coordenadora."),
            ("Às vezes, a solução é mais simples.", "As vezes, a solução é mais simples."),
            ("A empresa atendeu à demanda do mercado.", "A empresa atendeu a demanda do mercado."),
            ("O acesso à informação é um direito.", "O acesso a informação é um direito."),
            ("Compareceu à reunião sem atraso.", "Compareceu a reunião sem atraso."),
            ("À época, ninguém sabia o resultado.", "A época, ninguém sabia o resultado."),
            ("Devemos ir à farmácia agora.", "Devemos ir a farmácia agora."),
        ],
        "labels": ("com_crase", "sem_crase"),
    },
    "infinitivo_pessoal": {
        "name": "Infinitivo Pessoal",
        "type": "pairs",
        "pairs": [
            ("É importante para nós fazermos o trabalho.", "É importante para nós fazer o trabalho."),
            ("Antes de eles saírem de casa, verificaram tudo.", "Antes de eles sair de casa, verificaram tudo."),
            ("Apesar de nós termos estudado bastante.", "Apesar de nós ter estudado bastante."),
            ("Convém vocês pensarem bem antes de decidir.", "Convém vocês pensar bem antes de decidir."),
            ("É hora de nós partirmos para a viagem.", "É hora de nós partir para a viagem."),
            ("Sem eles perceberem, o tempo passou.", "Sem eles perceber, o tempo passou."),
            ("É necessário vocês trazerem os documentos.", "É necessário vocês trazer os documentos."),
            ("Antes de nós chegarmos, a festa começou.", "Antes de nós chegar, a festa começou."),
            ("Apesar de eles terem tentado, não conseguiram.", "Apesar de eles ter tentado, não conseguiram."),
            ("É melhor vocês ficarem em casa hoje.", "É melhor vocês ficar em casa hoje."),
            ("Para nós entendermos a lição, precisamos estudar.", "Para nós entender a lição, precisamos estudar."),
            ("Sem vocês ajudarem, não seria possível.", "Sem vocês ajudar, não seria possível."),
        ],
        "labels": ("flexionado", "não_flexionado"),
    },
    "cliticos": {
        "name": "Ordem dos Clíticos",
        "type": "pairs",
        "pairs": [
            ("Diga-me a verdade sobre o assunto.", "Me diga a verdade sobre o assunto."),
            ("Encontrou-se com o amigo na praça.", "Se encontrou com o amigo na praça."),
            ("Deram-lhe o prêmio de melhor aluno.", "Lhe deram o prêmio de melhor aluno."),
            ("Viu-o caminhando pela rua principal.", "O viu caminhando pela rua principal."),
            ("Apresentou-nos o novo projeto ontem.", "Nos apresentou o novo projeto ontem."),
            ("Contaram-me uma história interessante.", "Me contaram uma história interessante."),
            ("Entregou-lhes os documentos necessários.", "Lhes entregou os documentos necessários."),
            ("Chamaram-na para a entrevista final.", "A chamaram para a entrevista final."),
            ("Pediram-me para ficar até mais tarde.", "Me pediram para ficar até mais tarde."),
            ("Ofereceu-lhe um cargo na empresa.", "Lhe ofereceu um cargo na empresa."),
            ("Convidaram-nos para o jantar de sábado.", "Nos convidaram para o jantar de sábado."),
            ("Levou-a ao hospital imediatamente.", "A levou ao hospital imediatamente."),
            ("Mandaram-lhe uma mensagem urgente.", "Lhe mandaram uma mensagem urgente."),
            ("Trouxeram-me o pedido com atraso.", "Me trouxeram o pedido com atraso."),
        ],
        "labels": ("ênclise", "próclise"),
    },
}

# Load selected feature indices (same 150 as in the paper)
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./")
try:
    phase3 = torch.load(os.path.join(checkpoint_dir, "phase3_results.pt"), weights_only=False)
    all_selected = phase3["selected"]["all"]
except FileNotFoundError:
    try:
        phase4 = torch.load(os.path.join(checkpoint_dir, "phase4_probes_results.pt"), weights_only=False)
        all_selected = phase4["all_selected"]
    except FileNotFoundError:
        print("AVISO: Não encontrou phase3_results.pt nem phase4_probes_results.pt.")
        print("Usando top features conhecidas do paper como fallback.")
        all_selected = list(set(
            list(FEATURES_PT.values()) +
            [fid for fids in FEATURES_REGISTER.values() for fid in fids] +
            [10496, 1906, 8718, 16057, 12649, 16194, 657, 10126, 2909, 3623]
        ))

all_feature_indices = torch.tensor(all_selected)
n_features_selected = len(all_selected)
print(f"Features selecionadas: {n_features_selected}")

# Process probes
print("\nProcessando probes expandidos...")
t0 = time.time()

probe_raw_diffs = {}
probe_results_detail = {}

for phenom_key, phenom_data in EXPANDED_PROBES.items():
    print(f"  {phenom_data['name']}...", end=" ")

    if phenom_data["type"] == "pairs":
        pair_diffs = []
        for pair_a, pair_b in phenom_data["pairs"]:
            r_a = process_text(model, sae, pair_a, HOOK_NAME, all_feature_indices)
            r_b = process_text(model, sae, pair_b, HOOK_NAME, all_feature_indices)
            diff = (r_a["max_acts"] - r_b["max_acts"]).numpy()
            pair_diffs.append(diff)

        pair_diffs = np.array(pair_diffs)  # (n_pairs, n_features)
        probe_raw_diffs[phenom_key] = pair_diffs

        mean_diff = np.mean(np.abs(pair_diffs), axis=0)
        max_diff = np.max(np.abs(pair_diffs.mean(axis=0)))
        print(f"{len(phenom_data['pairs'])} pares | mean|Δ|={mean_diff.mean():.4f} | max|Δ|={max_diff:.4f}")

    elif phenom_data["type"] == "sets":
        set_variances = []
        for s in phenom_data["sets"]:
            set_acts = []
            for text in s:
                r = process_text(model, sae, text, HOOK_NAME, all_feature_indices)
                set_acts.append(r["max_acts"].numpy())
            set_acts = np.array(set_acts)  # (n_variants, n_features)
            set_variances.append(np.var(set_acts, axis=0))

        set_variances = np.array(set_variances)  # (n_sets, n_features)
        probe_raw_diffs[phenom_key] = set_variances

        mean_var = np.mean(set_variances, axis=0)
        print(f"{len(phenom_data['sets'])} sets | mean_var={mean_var.mean():.4f} | max_var={mean_var.max():.4f}")

elapsed = time.time() - t0
print(f"\nProbes processados em {elapsed:.0f}s")

# Bootstrap CI
print("\nBootstrap (1000 reamostras)...")
N_BOOTSTRAP = 1000
bootstrap_results = {}

# Re-seed numpy imediatamente antes do bootstrap para isolá-lo de qualquer
# consumo de aleatoriedade anterior (garante CIs idênticos entre execuções)
np.random.seed(GLOBAL_SEED)
for phenom_key, raw_diffs in probe_raw_diffs.items():
    n_samples = raw_diffs.shape[0]
    boot_means = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_sample = raw_diffs[idx]
        if EXPANDED_PROBES[phenom_key]["type"] == "pairs":
            boot_means.append(np.abs(boot_sample).mean())
        else:
            boot_means.append(boot_sample.mean())

    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    mean_val = np.mean(boot_means)

    if EXPANDED_PROBES[phenom_key]["type"] == "pairs":
        max_diff = float(np.max(np.abs(raw_diffs.mean(axis=0))))
    else:
        max_diff = float(np.max(raw_diffs.mean(axis=0)))

    bootstrap_results[phenom_key] = {
        "n_probes": n_samples,
        "mean": float(mean_val),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "max_diff": max_diff,
    }
    print(f"  {phenom_key:>20}: mean={mean_val:.4f} CI=[{ci_lo:.4f}, {ci_hi:.4f}] max={max_diff:.4f}")

ALL_RESULTS["expanded_probes"] = bootstrap_results
ALL_RESULTS["expanded_probes_config"] = {
    k: {"n_probes": len(v.get("pairs", v.get("sets", []))),
        "type": v["type"]}
    for k, v in EXPANDED_PROBES.items()
}

print(f"\nTotal de probes: {sum(v['n_probes'] for v in bootstrap_results.values())}")


# =========================================================================
# BLOCO 3: Multi-Feature Steering
# =========================================================================
print("\n" + "=" * 70)
print("BLOCO 3: MULTI-FEATURE STEERING")
print("=" * 70)

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

MULTI_PROMPTS = [
    "A menina bonita foi à",
    "O governo anunciou novas medidas para",
    "A pesquisa científica demonstrou que",
    "No fim de semana passado, a gente foi",
]

STEERING_CONFIGS = {
    "baseline": {"feature_ids": [], "multipliers": []},
    "gender_only": {
        "feature_ids": [FEATURES_PT["gender"]],
        "multipliers": [0.0],
    },
    "crase_only": {
        "feature_ids": [FEATURES_PT["crase"]],
        "multipliers": [0.0],
    },
    "clitics_only": {
        "feature_ids": [FEATURES_PT["enclisis"], FEATURES_PT["proclisis"]],
        "multipliers": [0.0, 0.0],
    },
    "all_morphosyntactic_ablated": {
        "feature_ids": [
            FEATURES_PT["gender"], FEATURES_PT["crase"],
            FEATURES_PT["enclisis"], FEATURES_PT["proclisis"],
        ],
        "multipliers": [0.0, 0.0, 0.0, 0.0],
    },
    "legal_amplified": {
        "feature_ids": FEATURES_REGISTER["legal"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["legal"]),
    },
    "journalistic_amplified": {
        "feature_ids": FEATURES_REGISTER["journalistic"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["journalistic"]),
    },
    "legal_vs_journalistic_conflict": {
        "feature_ids": FEATURES_REGISTER["legal"] + FEATURES_REGISTER["journalistic"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["legal"]) + [8.0] * len(FEATURES_REGISTER["journalistic"]),
    },
}

multi_feature_results = []

for config_name, config in STEERING_CONFIGS.items():
    print(f"\n  Config: {config_name}")
    config_data = {"config": config_name, "prompts": []}

    for prompt in MULTI_PROMPTS:
        if config_name == "baseline":
            text = generate_baseline(model, tokenizer, prompt,
                                     max_new_tokens=60, temperature=0.3)
        else:
            text = generate_with_steering(model, sae, tokenizer, prompt,
                                          feature_ids=config["feature_ids"],
                                          multipliers=config["multipliers"],
                                          max_new_tokens=60, temperature=0.3)

        acts = measure_feature_activations(
            model, sae, tokenizer, text,
            list(FEATURES_PT.values()) + ALL_REGISTER_FEATURE_IDS,
        )

        config_data["prompts"].append({
            "prompt": prompt,
            "generated": text,
            "activations": {str(k): v for k, v in acts.items() if isinstance(k, int)},
        })
        print(f"    {prompt[:30]}... → {text[len(prompt):len(prompt)+50]}...")

    multi_feature_results.append(config_data)

ALL_RESULTS["multi_feature_steering"] = multi_feature_results


# =========================================================================
# BLOCO 4: Register Steering Quantitativo
# =========================================================================
print("\n" + "=" * 70)
print("BLOCO 4: REGISTER STEERING QUANTITATIVO")
print("=" * 70)

REGISTER_PROMPTS = [
    "O governo anunciou novas medidas para",
    "A pesquisa sobre inteligência artificial",
    "O acidente na rodovia causou",
    "A empresa decidiu investir em",
]

REGISTER_CONDITIONS = {
    "baseline": {"feature_ids": [], "multipliers": []},
    "legal": {
        "feature_ids": FEATURES_REGISTER["legal"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["legal"]),
    },
    "scientific": {
        "feature_ids": FEATURES_REGISTER["scientific"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["scientific"]),
    },
    "colloquial": {
        "feature_ids": FEATURES_REGISTER["colloquial"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["colloquial"]),
    },
    "journalistic": {
        "feature_ids": FEATURES_REGISTER["journalistic"],
        "multipliers": [8.0] * len(FEATURES_REGISTER["journalistic"]),
    },
}

register_quant_results = {}

for reg_name, reg_config in REGISTER_CONDITIONS.items():
    print(f"\n  Register: {reg_name}")
    condition_acts = {fid: [] for fid in ALL_REGISTER_FEATURE_IDS}

    for prompt in REGISTER_PROMPTS:
        torch.manual_seed(42)
        if device == "cuda":
            torch.cuda.manual_seed_all(42)

        if reg_name == "baseline":
            text = generate_baseline(model, tokenizer, prompt,
                                     max_new_tokens=80, temperature=0.3)
        else:
            text = generate_with_steering(model, sae, tokenizer, prompt,
                                          feature_ids=reg_config["feature_ids"],
                                          multipliers=reg_config["multipliers"],
                                          max_new_tokens=80, temperature=0.3)

        acts = measure_feature_activations(
            model, sae, tokenizer, text, ALL_REGISTER_FEATURE_IDS,
        )

        for fid in ALL_REGISTER_FEATURE_IDS:
            condition_acts[fid].append(acts[fid]["mean"])

        print(f"    {prompt[:30]}... → done")

    register_quant_results[reg_name] = {}
    for fid in ALL_REGISTER_FEATURE_IDS:
        vals = condition_acts[fid]
        register_quant_results[reg_name][str(fid)] = {
            "mean_activation": float(np.mean(vals)),
            "std_activation": float(np.std(vals)),
        }

# Compute shift scores (steered - baseline)
print("\n  Register Shift Scores (steered - baseline):")
print(f"  {'Register':>15} | {'Target features':>20} | {'Shift score':>12} | {'Std':>8}")
print("  " + "-" * 65)

register_shift_scores = {}
for reg_name in ["legal", "scientific", "colloquial", "journalistic"]:
    target_fids = FEATURES_REGISTER[reg_name]
    shifts = []
    for fid in target_fids:
        baseline_mean = register_quant_results["baseline"][str(fid)]["mean_activation"]
        steered_mean = register_quant_results[reg_name][str(fid)]["mean_activation"]
        shifts.append(steered_mean - baseline_mean)

    mean_shift = float(np.mean(shifts))
    std_shift = float(np.std(shifts)) if len(shifts) > 1 else 0.0

    register_shift_scores[reg_name] = {
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "target_features": target_fids,
    }
    print(f"  {reg_name:>15} | {str(target_fids):>20} | {mean_shift:>+12.4f} | {std_shift:>8.4f}")

ALL_RESULTS["register_quantitative"] = register_quant_results
ALL_RESULTS["register_shift_scores"] = register_shift_scores


# =========================================================================
# SAVE ALL RESULTS
# =========================================================================
print("\n" + "=" * 70)
print("SALVANDO RESULTADOS")
print("=" * 70)

output_path = os.path.join(SAVE_DIR, "exp_robustness_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ALL_RESULTS, f, ensure_ascii=False, indent=2)

print(f"Resultados salvos em: {output_path}")
print(f"Tamanho: {os.path.getsize(output_path) / 1024:.1f} KB")

# Summary
print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)
print(f"  Bloco 1: {len(SEEDS)} seeds × {len(GENDER_PROMPTS)} prompts de gênero")
print(f"  Bloco 2: {sum(v['n_probes'] for v in bootstrap_results.values())} probes expandidos com CI 95%")
print(f"  Bloco 3: {len(STEERING_CONFIGS)} configurações de multi-feature steering")
print(f"  Bloco 4: {len(REGISTER_CONDITIONS)} condições de registro × {len(REGISTER_PROMPTS)} prompts")
print("\nPronto!")
