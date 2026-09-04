"""Step 7 = C2: encoding X -> M -- GATE G2

    /opt/conda/envs/talktuner-gpu/bin/python src/c2_encoding.py

Can credulity be linearly decoded from the residual stream of the COMBINED stimulus (stated
persona + behavioural conversation together -- the condition Gate G5 licensed, see WORKLOG entry
38)? And does that decoding beat what a dumb text classifier achieves on the same labels -- the
question that actually matters, since a probe that just beats chance is not evidence of anything
(DESIGN.md Sec 9).

Extracts the residual stream once per stimulus (data/c2_stimuli.json, built by
build_combined_stimuli.py) at the last token of [system turn]+[conversation], all 41 layers
(embedding + 40 blocks). Trains a per-layer logistic-regression probe, credulous vs skeptical,
train/held-out split by CONVERSATION IDENTITY (not item -- there are no items here).

Four baselines, all evaluated on the identical train/held-out split:
  - surface features (token/word/turn counts)      -- kills "probe is a verbosity detector" (T1)
  - TF-IDF on the stimulus text                     -- kills "probe is a lexical classifier" (T2)
  - layer-0 (embedding) probe                       -- kills "any layer would show this"
  - shuffled labels                                 -- kills leakage / extraction bugs

Gate G2 (GATES.md): best-layer probe accuracy >= best baseline accuracy + 5 points.

Also builds and saves the directions Steps 8-9 (C3/C4) will need: diff-in-means, probe weights,
a verbosity control (median length-split WITHIN each class), and a random control -- plus each
direction's cosine with the verbosity direction, so a high cosine is visible rather than assumed
away.
"""

import json
import re
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# n_train=336, d_model=5120 -- a heavily underdetermined regime (p >> n) where an
# under-regularized probe can partially fit even RANDOM labels. Regularization strength is
# chosen by 5-fold CV on the training set only (never touching held-out), independently per
# layer. Found this mattered in practice: with sklearn's default C=1.0, the shuffled-label
# control reached 76% instead of the expected ~50%, and cross-validating C brought it back down
# (WORKLOG entry 39/40) -- an unregularized probe is not a fair reading of what a layer encodes.
CV_CS = np.logspace(-4, 1, 12)

from subject_model import encode_prompt, load_subject, read_residual

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = REPO_ROOT / "data", REPO_ROOT / "outputs"
SUBJECT_MODEL = "NousResearch/Llama-2-13b-chat-hf"
SEED = 0
G2_MARGIN = 5.0  # percentage points


def stimulus_messages(s):
    return [{"role": "system", "content": s["template_text"]}] + s["turns"]


def stimulus_text(s):
    """All the text a surface/lexical baseline is allowed to see -- the persona sentence plus
    every conversation turn, concatenated. Same text the probe's activations were computed from."""
    return s["template_text"] + " " + " ".join(t["content"] for t in s["turns"])


def surface_features(s):
    convo_words = [w for t in s["turns"] for w in t["content"].split()]
    user_words = [w for t in s["turns"] if t["role"] == "user" for w in t["content"].split()]
    asst_words = [w for t in s["turns"] if t["role"] == "assistant" for w in t["content"].split()]
    text = stimulus_text(s)
    return [
        len(text.split()),                       # total words (proxy for total_tokens)
        text.count("?"),                         # question marks
        len(user_words),
        len(asst_words),
        len(s["turns"]),                         # n_turns
        len(s["template_text"].split()),         # persona-sentence length on its own
    ]


def extract_activations(stimuli, model, tok, batch_size=8):
    """One forward pass per stimulus. Returns (n_stimuli, n_layers+1, d_model) float32 array."""
    ids_list, pos_list = [], []
    for s in stimuli:
        msgs = stimulus_messages(s)
        last_user = max(i for i, t in enumerate(msgs) if t["role"] == "user")
        ids = encode_prompt(tok, msgs[:last_user + 1], add_generation_prompt=True)
        ids_list.append(ids)
        pos_list.append(len(ids) - 1)
    acts = read_residual(model, tok, ids_list, pos_list, batch_size=batch_size)
    return acts.numpy()


def probe_accuracy_by_layer(X_train, y_train, X_held, y_held, seed=SEED, return_clfs=False):
    """LogisticRegressionCV per layer -- C selected by 5-fold CV on X_train/y_train only, so the
    held-out set never influences regularization strength. X is (n, n_layers+1, d_model)."""
    accs, clfs = [], []
    n_layers = X_train.shape[1]
    for L in range(n_layers):
        clf = LogisticRegressionCV(Cs=CV_CS, cv=5, penalty="l2", max_iter=3000,
                                   random_state=seed, n_jobs=-1)
        clf.fit(X_train[:, L, :], y_train)
        accs.append(clf.score(X_held[:, L, :], y_held))
        clfs.append(clf)
    return (accs, clfs) if return_clfs else accs


def shuffled_label_baseline(X_train, y_train, X_held, y_held, seed=SEED):
    """Full per-layer report, not just the max -- max-of-41-layers on pure noise is itself a
    multiple-comparisons inflation and conflating it with genuine signal is a second error on
    top of under-regularization. Both are worth seeing separately."""
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y_train)
    accs = probe_accuracy_by_layer(X_train, y_shuf, X_held, y_held, seed=seed)
    return accs


def main():
    stimuli = json.load(open(DATA / "c2_stimuli.json"))
    # Binary credulous/skeptical for the primary probe -- neutral held separately (used later for
    # C3/C4's mu, the mean projection under neutral).
    binary = [s for s in stimuli if s["label"] in ("credulous", "skeptical")]
    train = [s for s in binary if s["split"] == "train"]
    held = [s for s in binary if s["split"] == "held_out"]
    print("Binary (credulous/skeptical) stimuli: %d train, %d held-out" % (len(train), len(held)))

    model, tok = load_subject(SUBJECT_MODEL, dtype=torch.bfloat16, device="cuda")
    print("Loaded %s | %.1f GB allocated\n" % (SUBJECT_MODEL, torch.cuda.memory_allocated() / 1e9))

    import time
    t0 = time.time()
    X_all = extract_activations(stimuli, model, tok)
    print("Extracted activations for %d stimuli (all classes) in %.1f min, shape %s"
          % (len(stimuli), (time.time() - t0) / 60, X_all.shape))

    id_to_idx = {s["stimulus_id"]: i for i, s in enumerate(stimuli)}
    X_train = X_all[[id_to_idx[s["stimulus_id"]] for s in train]]
    X_held = X_all[[id_to_idx[s["stimulus_id"]] for s in held]]
    y_train = np.array([s["label"] == "credulous" for s in train], dtype=int)
    y_held = np.array([s["label"] == "credulous" for s in held], dtype=int)

    n_layers = X_all.shape[1]

    # -------------------------------------------------------------- probe accuracy by layer
    probe_accs, probe_clfs = probe_accuracy_by_layer(X_train, y_train, X_held, y_held, return_clfs=True)
    best_layer = int(np.argmax(probe_accs))
    best_acc = probe_accs[best_layer]
    print("\nProbe accuracy by layer (held-out, n=%d, C chosen by 5-fold CV on train only):" % len(held))
    for L in range(0, n_layers, 4):
        print("  layer %2d: %.3f  (C=%.4g)" % (L, probe_accs[L], probe_clfs[L].C_[0]))
    print("  BEST layer %d: %.3f" % (best_layer, best_acc))

    # -------------------------------------------------------------- shuffled-label noise floor
    # Full per-layer report FIRST, so "the noise floor" is read off the data, not assumed to be
    # 0.5 -- max-of-41-layers-on-noise and under-regularization both inflate this if not checked.
    print("\nShuffled-label control, per layer (labels shuffled in TRAIN only; expect ~0.5):")
    shuf_accs = shuffled_label_baseline(X_train, y_train, X_held, y_held)
    for L in range(0, n_layers, 4):
        print("  layer %2d: %.3f" % (L, shuf_accs[L]))
    shuf_at_best_layer = shuf_accs[best_layer]
    shuf_max = max(shuf_accs)
    print("  at the REAL best layer (%d): %.3f   |   max over all 41 layers: %.3f"
          % (best_layer, shuf_at_best_layer, shuf_max))
    if shuf_max > 0.60:
        print("  NOTE: noise floor is still above chance even after CV regularization -- report")
        print("  the max, not 0.5, as what 'beating chance' has to clear.")

    # -------------------------------------------------------------- other baselines
    print("\nOther baselines (held-out accuracy):")
    surf_train = np.array([surface_features(s) for s in train])
    surf_held = np.array([surface_features(s) for s in held])
    surf_clf = LogisticRegressionCV(Cs=CV_CS, cv=5, max_iter=3000, random_state=SEED, n_jobs=-1)
    surf_clf.fit(surf_train, y_train)
    surf_acc = surf_clf.score(surf_held, y_held)
    print("  surface features:   %.3f" % surf_acc)

    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_train = tfidf.fit_transform([stimulus_text(s) for s in train])
    tfidf_held = tfidf.transform([stimulus_text(s) for s in held])
    tfidf_clf = LogisticRegressionCV(Cs=CV_CS, cv=5, max_iter=3000, random_state=SEED, n_jobs=-1)
    tfidf_clf.fit(tfidf_train, y_train)
    tfidf_acc = tfidf_clf.score(tfidf_held, y_held)
    print("  TF-IDF:             %.3f" % tfidf_acc)

    embed_acc = probe_accs[0]
    print("  layer-0 (embedding): %.3f" % embed_acc)

    # The honest baseline the probe must clear is the BEST of: the three text/surface baselines,
    # AND the shuffled-label noise ceiling (whichever of "max over layers" or "at this layer" is
    # higher, i.e. the more conservative, harder-to-beat reading).
    best_baseline = max(surf_acc, tfidf_acc, embed_acc, shuf_max)
    margin = 100 * (best_acc - best_baseline)

    print("\n" + "=" * 70)
    print("GATE G2: best-layer accuracy >= best baseline + %.0f points" % G2_MARGIN)
    print("  best probe layer %d: %.1f%%   best baseline (incl. shuffled-noise ceiling): %.1f%%   "
          "margin: %+.1f points" % (best_layer, 100 * best_acc, 100 * best_baseline, margin))
    passed = margin >= G2_MARGIN
    print("  RESULT: %s" % ("PASS" if passed else
          "FAIL -- reframe as 'credulity probes are text classifiers' (GATES.md outcome table)"))
    print("=" * 70)

    # -------------------------------------------------------------- directions (for C3/C4)
    h_cred = X_train[y_train == 1, best_layer, :]
    h_skep = X_train[y_train == 0, best_layer, :]
    diff_means = h_cred.mean(axis=0) - h_skep.mean(axis=0)
    diff_means = diff_means / np.linalg.norm(diff_means)

    probe_best = probe_clfs[best_layer]  # already fit, C chosen by CV -- reuse, don't refit unregularized
    probe_dir = probe_best.coef_[0] / np.linalg.norm(probe_best.coef_[0])

    # Verbosity control: median-split by stimulus TEXT LENGTH, WITHIN each class -- by
    # construction this direction tracks length, not credulity (DESIGN.md Sec 7).
    lengths = np.array([len(stimulus_text(s).split()) for s in train])
    verb_diff = np.zeros_like(diff_means)
    for cls_val in (0, 1):
        mask = y_train == cls_val
        med = np.median(lengths[mask])
        long_h = X_train[mask & (lengths > med), best_layer, :]
        short_h = X_train[mask & (lengths <= med), best_layer, :]
        if len(long_h) and len(short_h):
            verb_diff += (long_h.mean(axis=0) - short_h.mean(axis=0))
    verb_dir = verb_diff / np.linalg.norm(verb_diff)

    rng = np.random.default_rng(SEED)
    random_dir = rng.standard_normal(diff_means.shape[0]).astype(np.float32)
    random_dir /= np.linalg.norm(random_dir)

    ortho_dir = diff_means - float(diff_means @ verb_dir) * verb_dir
    ortho_dir /= np.linalg.norm(ortho_dir)

    cos_dm_verb = float(diff_means @ verb_dir)
    print("\ncos(diff-in-means, verbosity direction) at best layer: %+.3f" % cos_dm_verb)

    np.savez(DATA / "c2_directions.npz",
             best_layer=best_layer,
             diff_in_means=diff_means, probe_weights=probe_dir,
             verbosity=verb_dir, random=random_dir, orthogonalized=ortho_dir)
    print("Saved directions -> data/c2_directions.npz")

    json.dump({
        "best_layer": best_layer,
        "probe_accuracy_by_layer": [float(a) for a in probe_accs],
        "shuffled_accuracy_by_layer": [float(a) for a in shuf_accs],
        "shuffled_at_best_layer": float(shuf_at_best_layer), "shuffled_max": float(shuf_max),
        "baselines": {"surface_features": float(surf_acc), "tfidf": float(tfidf_acc),
                       "embedding_layer0": float(embed_acc), "shuffled_labels_max": float(shuf_max)},
        "best_baseline": float(best_baseline), "margin_points": float(margin),
        "gate_g2_passed": bool(passed),
        "cos_diff_in_means_verbosity": float(cos_dm_verb),
        "n_train": len(train), "n_held_out": len(held),
    }, open(DATA / "c2_summary.json", "w"), indent=2)
    print("Saved summary -> data/c2_summary.json")


if __name__ == "__main__":
    main()
