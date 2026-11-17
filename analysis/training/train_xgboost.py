"""Train and evaluate a first, event-separated BIB/LLP XGBoost baseline.

This is a comparison of one BIB-enriched data sample and one LLP simulation,
inside a common selection. It is not a calibrated probability or a measurement
of detector-wide BIB rejection. All thresholds are chosen on validation only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from analysis import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "analysis/.cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from analysis.data.xgb_data import (
    KINEMATIC_FEATURES, PHYSICS_FEATURES, physical_invalid_mask, prepare_training_data, sanitize_values, tile_fraction_mask,
)
from analysis.evaluation.xgb_evaluation import DELAYED_TIME_NS, choose_threshold, evaluate_scores, make_plots
from analysis.evaluation.xgb_matching import _bin_indices, match_kinematics


MODEL_FEATURES = {
    "XGBoost": list(PHYSICS_FEATURES),
    "Timing only": ["jet_time_ATLAS", "jet_time_spread"],
    "Kinematics only": list(KINEMATIC_FEATURES),
    "Without time uncertainty": [f for f in PHYSICS_FEATURES if f != "jet_time_sigma"],
}
MODEL_SLUGS = {name: name.lower().replace(" ", "_") for name in MODEL_FEATURES}
PREDICTION_COLUMNS = ["label", "sample", "event_group", "source_file", "row_index", "jet_index",
                      "jet_pt_GeV", "jet_eta", "jet_time_ATLAS"]
EARLY_STOP_ROUNDS = 40
UPSTREAM = {
    "url": "https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public",
    "inspected_commit": "a7c625c364893bbc90248fbb98a5251c1a54fdeb",
    "production_commit_confirmed": False,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def balanced_weights(labels, reference_labels):
    """Use training class counts only; these are not luminosity/MC weights."""
    reference_labels = np.asarray(reference_labels, dtype=int)
    counts = np.bincount(reference_labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training needs jets in both classes.")
    factors = len(reference_labels) / (2 * counts)
    return factors[np.asarray(labels, dtype=int)]


def predict_model(model, frame, features):
    # Float64 is deliberate: a nextafter threshold must keep tied float32 scores.
    return model.predict_proba(frame[list(features)])[:, 1].astype(np.float64)


def fit_booster(train, val, features, parameters, rounds, train_weights, val_weights, *, include_train_eval=False):
    """Fit with early stopping on validation; optionally record the training history."""
    features = list(features)
    train_xy = train[features], train["label"]
    eval_set = [(val[features], val["label"])]
    eval_weights = [val_weights]
    if include_train_eval:
        eval_set.insert(0, train_xy)
        eval_weights.insert(0, train_weights)
    model = XGBClassifier(**parameters, callbacks=[EarlyStopping(
        rounds=rounds, metric_name="logloss", data_name=f"validation_{len(eval_set) - 1}", save_best=True,
    )])
    model.fit(*train_xy, sample_weight=train_weights, eval_set=eval_set,
              sample_weight_eval_set=eval_weights, verbose=False)
    return model


def _save_and_verify_model(model, path, frame, features, expected_scores, error):
    """Ensure the saved booster reproduces the validation predictions exactly."""
    model.save_model(path)
    reloaded = XGBClassifier()
    reloaded.load_model(path)
    if not np.array_equal(expected_scores, predict_model(reloaded, frame, features)):
        raise AssertionError(error)


def score_saved_model(frame, output_dir):
    """Score selected, flattened jet rows with the saved main model.

    The output includes an acceptance mask: out-of-selection jets have NaN
    scores, rather than silently receiving an unsupported decision. Required
    momenta are in GeV, eta is dimensionless, and stored timing is in ns.
    """
    output_dir = Path(output_dir)
    spec = json.loads((output_dir / "feature_schema.json").read_text())
    features = spec["features"]
    required = set(features) | {"jet_pt_GeV", "jet_eta"}
    if required - set(frame.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(frame.columns))}")
    x = frame.copy()
    numeric = list(required)
    x[numeric] = x[numeric].astype(float).replace([-999999.0, np.inf, -np.inf], np.nan)
    for field in features:
        values = sanitize_values(x[field].to_numpy())
        values[physical_invalid_mask(field, values)] = np.nan
        x[field] = values
    sel = spec["selection"]
    valid = (
        x["jet_pt_GeV"].ge(sel["min_pt_GeV"])
        & x["jet_eta"].abs().lt(sel["max_abs_eta"])
        & np.isfinite(x["jet_eta"])
        & np.isfinite(x["jet_time_ATLAS"])
        & tile_fraction_mask(x["jet_TileEnergyFrac"], require_mixed=True)
    )
    support = spec.get("kinematic_support")
    if support:
        pt_edges = np.asarray(support["pt_edges_GeV"])
        eta_edges = np.asarray(support["abs_eta_edges"])
        pt = x["jet_pt_GeV"].to_numpy()
        eta = x["jet_eta"].abs().to_numpy()
        pt_bin = _bin_indices(pt, pt_edges)
        eta_bin = _bin_indices(eta, eta_edges)
        bins = {(b["pt_bin"], b["eta_bin"]) for b in support["supported_bins"]}
        supported = np.array([(p, e) in bins for p, e in zip(pt_bin, eta_bin)], dtype=bool)
        valid &= (pt_bin >= 0) & (eta_bin >= 0) & supported
    model = XGBClassifier()
    model.load_model(output_dir / "xgboost.ubj")
    scores = np.full(len(x), np.nan)
    if valid.any():
        scores[np.flatnonzero(valid)] = predict_model(model, x.loc[valid], features)
    tags = pd.array([pd.NA] * len(x), dtype="boolean")
    tags[np.flatnonzero(valid)] = scores[valid] >= spec["threshold"]
    return pd.DataFrame({"in_selection": valid.to_numpy(), "bib_score": scores,
                         "tagged_as_bib": tags}, index=frame.index)


def _subset_report(frame, scores, threshold, seed):
    """Fixed diagnostic regions; never use these test results to select a cut."""
    regions = {
        "delayed_time_gt_3ns": frame["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy(),
        "pt_30_to_100_GeV": frame["jet_pt_GeV"].lt(100).to_numpy(),
        "pt_100_to_300_GeV": (frame["jet_pt_GeV"].ge(100) & frame["jet_pt_GeV"].lt(300)).to_numpy(),
        "pt_ge_300_GeV": frame["jet_pt_GeV"].ge(300).to_numpy(),
    }
    result = {}
    for name, mask in regions.items():
        result[name] = (evaluate_scores(frame.loc[mask], scores[mask], threshold,
                                       bootstrap_repeats=200, seed=seed)
                        if mask.any() else None)
    return result


def _fit_models(train, validation, params, target, output_dir, *, matched=False):
    """Fit each feature set, lock its validation cut, and verify the saved model."""
    train_w, val_w = (balanced_weights(frame.label, train.label) for frame in (train, validation))
    models, thresholds, histories, metrics, importance = {}, {}, {}, {}, []
    for name, features in MODEL_FEATURES.items():
        if not matched:
            print(f"Training {name}: {', '.join(features)}", flush=True)
        model = fit_booster(train, validation, features, params, EARLY_STOP_ROUNDS, train_w, val_w,
                            include_train_eval=True)
        val_scores = predict_model(model, validation, features)
        cut = choose_threshold(validation.label, val_scores, target)
        _save_and_verify_model(model, output_dir / f"{MODEL_SLUGS[name]}.ubj", validation, features, val_scores,
                               "Saved model predictions differ after loading.")
        models[name], thresholds[name], histories[name] = model, cut, model.evals_result()
        details = {"features": features}
        if not matched:
            details["best_iteration_zero_based"] = int(model.best_iteration)
        metrics[name] = dict(details, saved_trees=int(model.get_booster().num_boosted_rounds()),
                             validation=evaluate_scores(validation, val_scores, cut, bootstrap_repeats=0))
        importance.extend({"model": name, "feature": field, "importance": float(gain)}
                          for field, gain in zip(features, model.feature_importances_))
    return models, thresholds, histories, metrics, pd.DataFrame(importance)


def _test_models(test, models, thresholds, metrics, seed, *, matched=False):
    """Evaluate already fitted models with their frozen validation cuts."""
    scores = {}
    for name, model in models.items():
        score = scores[name] = predict_model(model, test, MODEL_FEATURES[name])
        cut = thresholds[name]
        stats = metrics[name]["test"] = evaluate_scores(test, score, cut, seed=seed)
        metrics[name]["test_regions"] = _subset_report(test, score, cut, seed)
        heading = f"Matched {name}: AUC" if matched else f"{name}: test AUC"
        print(f"{heading}={stats['roc_auc']:.4f}; BIB tag={stats['bib_tag_fraction']:.1%}, "
              f"LLP keep={stats['llp_keep_fraction']:.1%}", flush=True)
    return scores


def _write_predictions(test, scores, output_dir, *, thresholds=None):
    columns = PREDICTION_COLUMNS if thresholds is None else [*PREDICTION_COLUMNS, "jet_BIB_score"]
    predictions = test[columns].copy()
    for name, score in scores.items():
        slug = MODEL_SLUGS[name]
        predictions[f"{slug}_score"] = score
        if thresholds is not None:
            predictions[f"{slug}_tagged_as_bib"] = score >= thresholds[name]
    predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False)


def _plot_inputs(frame, features, output_dir):
    n = len(features)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 7.3), squeeze=False)
    labels = {
        "jet_time_ATLAS": "ATLAS jet time [ns]",
        "jet_time_spread": "Cell-time spread [ns]",
        "jet_time_sigma": "Jet time uncertainty [ns]",
        "jet_n_clusters": "Number of clusters",
        "jet_TileEnergyFrac": "Tile energy fraction",
    }
    for ax, feature in zip(axes.flat, features):
        values = frame[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [0.005, 0.995])
        if low == high:
            low, high = low - .5, high + .5
        bins = np.linspace(low, high, 36)
        for label, name, color in [(0, "Matched LLP", "tab:blue"), (1, "Unpaired", "tab:orange")]:
            v = frame.loc[frame.label == label, feature].dropna().to_numpy()
            ax.hist(v, bins=bins, weights=np.ones(len(v)) / int((frame.label == label).sum()),
                    histtype="step", linewidth=1.8, label=name, color=color)
        ax.set(xlabel=labels.get(feature, feature), ylabel="Fraction of all class jets / bin")
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Training inputs in the common analysis region")
    fig.text(.5, .01, "Training split only; ranges show central 99% of pooled values. Tails are omitted, not renormalized.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    path = output_dir / "training_inputs.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _reference_comparison(test, validation, model_scores, main_threshold, target, output_dir, seed):
    """Compare with the stored fit score only where that score is available."""
    test_mask = test.jet_BIB_score.notna().to_numpy()
    val_mask = validation.jet_BIB_score.notna().to_numpy()
    report = {
        "test_coverage_by_class": {
            name: float(test_mask[test.label.to_numpy() == label].mean())
            for label, name in [(0, "LLP"), (1, "BIB")]
        },
        "definition": "Higher stored score is more BIB-like; -999999/nonfinite values are missing.",
        "fairness": "Both methods use the same valid-score validation and test subsets. This comparison has a separate validation cut; the primary model cut stays frozen elsewhere.",
    }
    if (not val_mask.any() or not test_mask.any()
            or validation.loc[val_mask].label.nunique() != 2
            or test.loc[test_mask].label.nunique() != 2):
        report["available"] = False
        return report, None
    legacy_cut = choose_threshold(validation.loc[val_mask, "label"],
                                  validation.loc[val_mask, "jet_BIB_score"], target)
    # Calibrate both operating points on the same validation population.
    main_model = XGBClassifier()
    main_model.load_model(output_dir / "xgboost.ubj")
    same_val_scores = predict_model(main_model, validation.loc[val_mask], MODEL_FEATURES["XGBoost"])
    same_main_cut = choose_threshold(validation.loc[val_mask, "label"], same_val_scores, target)
    same_test = test.loc[test_mask]
    main = model_scores[test_mask]
    legacy = test.loc[test_mask, "jet_BIB_score"].to_numpy()
    report.update({"available": True, "stored_score_threshold": legacy_cut,
                   "XGBoost_same_validation_threshold": same_main_cut,
                   "primary_XGBoost_threshold_for_reference": main_threshold,
                   "XGBoost_same_jets": evaluate_scores(same_test, main, same_main_cut, seed=seed),
                   "Stored_BIB_score_same_jets": evaluate_scores(same_test, legacy, legacy_cut, seed=seed)})
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6.7, 5.2))
    for name, scores, metric in [("XGBoost", main, report["XGBoost_same_jets"]),
                                  ("Stored BIB score", legacy, report["Stored_BIB_score_same_jets"])]:
        fpr, tpr, _ = roc_curve(same_test.label, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC {metric['roc_auc']:.3f})")
        ax.scatter(1 - metric["llp_keep_fraction"], metric["bib_tag_fraction"], s=35)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set(xlabel="Fraction of matched LLP jets tagged as BIB", ylabel="Fraction of unpaired jets tagged as BIB",
           title="Same test jets with a valid stored BIB score", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = output_dir / "stored_tagger_comparison.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return report, path


def _matched_check(frame, params, output_dir, *, seed, target, selection):
    """An exploratory conditional check, after the initial sample comparison.

    Kinematic matching never uses scores or timing and never moves an event
    between splits. It changes the evaluated population, so its rates are
    reported separately rather than as replacements for the original test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matched, audit = match_kinematics(frame, seed=seed)
    matched.to_csv(output_dir / "matched_jets.csv.gz", index=False)
    write_json(output_dir / "matching_audit.json", audit)
    train = matched.loc[matched.split == "train"]
    validation = matched.loc[matched.split == "validation"]
    test = matched.loc[matched.split == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, (data, title) in enumerate([(frame.loc[frame.split == "train"], "Original training sample"),
                                         (train, "After bin balancing")]):
        for col, (field, edges, xlabel) in enumerate([
                ("jet_pt_GeV", audit["pt_edges_GeV"], "Jet pT [GeV]"),
                ("jet_abs_eta", audit["abs_eta_edges"], "Absolute jet eta")]):
            for label, name, color in [(0, "Matched LLP", "tab:blue"), (1, "Unpaired", "tab:orange")]:
                values = data.loc[data.label == label, field].to_numpy()
                axes[row, col].hist(values, bins=edges, weights=np.ones(len(values)) / len(values),
                                    histtype="step", linewidth=1.8, label=name, color=color)
            axes[row, col].set(xlabel=xlabel, ylabel="Fraction of class jets / bin", title=title)
            axes[row, col].grid(alpha=.2)
            if col == 0:
                axes[row, col].set_xscale("log")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Training kinematics before and after balancing in joint bins")
    fig.tight_layout()
    matching_plot = output_dir / "kinematics_before_after.png"
    fig.savefig(matching_plot, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Kinematic check: {len(train):,} train, {len(validation):,} validation, {len(test):,} test jets.", flush=True)
    models, thresholds, histories, metrics, gain = _fit_models(train, validation, params, target, output_dir, matched=True)
    scores = _test_models(test, models, thresholds, metrics, seed, matched=True)
    plots = make_plots(test, scores, thresholds, histories, gain, output_dir)
    plots.append(matching_plot)
    _write_predictions(test, scores, output_dir)
    write_json(output_dir / "training_history.json", histories)
    write_json(output_dir / "feature_schema.json", {
        "features": MODEL_FEATURES["XGBoost"], "threshold": thresholds["XGBoost"],
        "selection": selection, "units": {"momentum": "GeV", "time": "ns", "eta": "dimensionless"},
        "tag_rule": "score >= threshold", "validation_target_llp_keep": target,
        "score_meaning": "BIB classifier score; not a calibrated physical probability",
        "training_population": "Kinematic-bin balanced subset of the common analysis region",
        "kinematic_support": {key: audit[key] for key in ["pt_edges_GeV", "abs_eta_edges", "supported_bins"]},
        "missing": "-999999, nonfinite and physically invalid timing/count values -> NaN",
    })
    saved_scores = score_saved_model(test, output_dir)
    if not saved_scores.in_selection.all() or not np.array_equal(saved_scores.bib_score.to_numpy(), scores["XGBoost"]):
        raise AssertionError("Matched model inference disagrees with experiment predictions.")
    report = {
        "scope": "Exploratory secondary check motivated by strong kinematic separation in the initial run; not an independent final validation.",
        "procedure": "Original event split retained. Train-supported fixed pT/|eta| bins, equal class counts per bin in each split. No timing or model-score matching.",
        "caveat": "Coarse-bin matching does not make the samples identical. Read the kinematics-only control and residual production/MC-data caveats.",
        "models": metrics, "matching_audit_file": "matching_audit.json",
        "plots": [p.name for p in plots], "n_selected_jets": len(matched),
    }
    write_json(output_dir / "metrics.json", report)
    return report


def run_experiment(project_root, output_dir=None, *, seed=42, target_llp_keep=0.95,
                   min_pt_GeV=30., max_abs_eta=1.7, verify_hashes=True):
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir or project_root / "analysis/outputs/xgboost_baseline").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("Reading the selected ROOT files and checking their provenance...", flush=True)
    frame, audit = prepare_training_data(project_root, output_dir, seed=seed,
                                         min_pt_GeV=min_pt_GeV, max_abs_eta=max_abs_eta,
                                         verify_hashes=verify_hashes)
    train = frame.loc[frame.split == "train"].copy()
    validation = frame.loc[frame.split == "validation"].copy()
    test = frame.loc[frame.split == "test"].copy()
    print(f"Selected {len(frame):,} jets: {len(train):,} train, {len(validation):,} validation, {len(test):,} test.", flush=True)
    params = dict(n_estimators=600, learning_rate=.05, max_depth=3, min_child_weight=10,
                  subsample=.8, colsample_bytree=1., reg_lambda=5., reg_alpha=0.,
                  objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  n_jobs=4, random_state=seed, base_score=.5)
    # Fixed settings for this reproducible run. The no-sigma ablation and
    # matching study are exploratory follow-ups to the initial three models.
    models, thresholds, histories, metrics, gain = _fit_models(train, validation, params, target_llp_keep, output_dir)
    scores = _test_models(test, models, thresholds, metrics, seed)
    # Validation-only permutation checks avoid using test importance to tune the model.
    perm = permutation_importance(models["XGBoost"], validation[MODEL_FEATURES["XGBoost"]],
                                  validation.label, scoring="roc_auc", n_repeats=8,
                                  random_state=seed, n_jobs=1)
    permutation = pd.DataFrame({"feature": MODEL_FEATURES["XGBoost"],
                                "auc_drop_mean": perm.importances_mean,
                                "auc_drop_std": perm.importances_std})
    permutation.to_csv(output_dir / "validation_permutation_importance.csv", index=False)
    gain.to_csv(output_dir / "feature_importance.csv", index=False)
    plots = make_plots(test, scores, thresholds, histories, gain, output_dir)
    plots.append(_plot_inputs(train, MODEL_FEATURES["XGBoost"], output_dir))
    comparison, plot = _reference_comparison(test, validation, scores["XGBoost"], thresholds["XGBoost"],
                                            target_llp_keep, output_dir, seed)
    if plot:
        plots.append(plot)
    matched_report = _matched_check(frame, params, output_dir / "matched_kinematics",
                                     seed=seed, target=target_llp_keep,
                                     selection={"min_pt_GeV": min_pt_GeV, "max_abs_eta": max_abs_eta,
                                                "tile_fraction": "strictly between 0 and 1"})
    _write_predictions(test, scores, output_dir, thresholds=thresholds)
    schema = {
        "features": MODEL_FEATURES["XGBoost"], "units": {"energy_momentum": "GeV after MeV/1000 conversion", "time": "ns", "eta": "dimensionless"},
        "label_definition": {"0": "truth-matched LLP jet in selected simulation", "1": "unpaired jet, working BIB label"},
        "selection": {"min_pt_GeV": min_pt_GeV, "max_abs_eta": max_abs_eta, "tile_fraction": "strictly between 0 and 1"},
        "threshold": thresholds["XGBoost"], "tag_rule": "score >= threshold", "validation_target_llp_keep": target_llp_keep,
        "score_meaning": "BIB classifier score; not a calibrated physical probability", "missing": "-999999 and nonfinite -> NaN",
    }
    write_json(output_dir / "feature_schema.json", schema)
    # Exercise the public, scope-checked inference path against the saved model.
    inference = score_saved_model(test, output_dir)
    if not inference.in_selection.all() or not np.array_equal(inference.bib_score.to_numpy(), scores["XGBoost"]):
        raise AssertionError("Saved inference wrapper disagrees with experiment predictions.")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "target_llp_keep_on_validation": target_llp_keep, "selection": schema["selection"],
        "data_audit_file": "data_audit.json", "upstream": UPSTREAM,
        "models": metrics, "stored_tagger_comparison": comparison,
        "matched_kinematics": matched_report,
        "training_parameters": params, "early_stopping": {"rounds": EARLY_STOP_ROUNDS, "metric": "class-balanced validation logloss", "test_used": False},
        "study_sequence": "Initial five-input, timing-only and kinematics-only models used fixed settings. After inspecting that comparison, bin balancing and the no-time-uncertainty ablation were added as exploratory diagnostics. No hyperparameter tuning or final model selection on the test was performed; further improvements need a fresh independent test.",
        "weighting": "Training and early stopping use inverse training-class-frequency weights. Reported test jet fractions are unweighted and are not event rates.",
        "uncertainty": "Event-bootstrap test-rate intervals hold fitted model and validation threshold fixed. They exclude training/threshold/purity and data-MC uncertainty.",
        "limitations": [
            "Unpaired is a working BIB label and can contain collision contamination; no n_vrtx branch is available for cleaning.",
            "LLP truth matching is defined by upstream source; exact ntuple production commit still needs confirmation with Emil.",
            "Both classes require mixed LAr/Tile calorimeter support. Results do not apply to excluded pure-calorimeter jets.",
            "One LLP mass/lifetime and one unpaired run; event-separated test is not an independent run, mass or production test.",
            "Signal and background have different momentum spectra and MC/data production. Removing explicit pT/eta does not remove all kinematic dependence.",
            "No ordinary-collision evaluation: supplied ZMUMU files are AOD CollectionTree, not the JetContent ntuple used here.",
            "No luminosity, trigger-efficiency or MC event weights; no detector-wide rejection or physical probability claim.",
        ],
        "plots": [str(Path(p).relative_to(output_dir)) for p in plots],
        "versions": {pkg: version(pkg) for pkg in ["xgboost", "scikit-learn", "numpy", "pandas", "uproot", "awkward", "matplotlib", "scipy"]},
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), PROJECT_ROOT / "analysis/data/xgb_data.py", PROJECT_ROOT / "analysis/evaluation/xgb_evaluation.py", PROJECT_ROOT / "analysis/evaluation/xgb_matching.py"]},
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_json(output_dir / "training_history.json", histories)
    write_json(output_dir / "metrics.json", report)
    summary_rows = [{"model": name, "n_test_jets": m["test"]["n_jets"],
                     "roc_auc": m["test"]["roc_auc"], "average_precision": m["test"]["average_precision"],
                     "BIB_tag_fraction": m["test"]["bib_tag_fraction"], "LLP_keep_fraction": m["test"]["llp_keep_fraction"],
                     "threshold_from_validation": thresholds[name], "trees": m["saved_trees"]}
                    for name, m in metrics.items()]
    pd.DataFrame(summary_rows).to_csv(output_dir / "results.csv", index=False)
    _write_report(report, output_dir)
    print(f"Saved models, metrics, report and plots to {output_dir}", flush=True)
    return report


def _write_report(report, output_dir):
    selection = report["selection"]
    target = report["target_llp_keep_on_validation"]
    lines = [
        "# XGBoost baseline", "",
        f"Unpaired and truth-matched LLP jets with pT >= {selection['min_pt_GeV']:g} GeV, "
        f"|eta| < {selection['max_abs_eta']:g} and 0 < Tile energy fraction < 1.",
        f"Cuts target {target:.0%} LLP retention on validation. All jets in an event share one split.", "",
    ]
    samples = {"Original sample": report["models"], "Shared pT and |eta|": report["matched_kinematics"]["models"]}
    for title, models in samples.items():
        lines += [f"## {title}", "", "| Model | AUC | Unpaired tagged | LLP kept |", "| --- | ---: | ---: | ---: |"]
        for name, model in models.items():
            m = model["test"]
            lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['bib_tag_fraction']:.1%} | {m['llp_keep_fraction']:.1%} |")
        delayed = models["XGBoost"]["test_regions"]["delayed_time_gt_3ns"]
        if delayed:
            counts, decisions = delayed["class_counts"], delayed["confusion"]
            lines += ["", f"Above 3 ns, XGBoost tagged {decisions['BIB_tagged_BIB']}/{counts['BIB']} unpaired jets "
                      f"and kept {decisions['LLP_kept_as_LLP']}/{counts['LLP']} LLP jets."]
        lines.append("")
    lines += ["## Limits", "", *[f"- {item}" for item in report["limitations"]], "",
              "See metrics.json for counts and event-bootstrap intervals, data_audit.json for input checks, "
              "and feature_schema.json beside each saved model for its inputs and cut.", ""]
    (output_dir / "run_notes.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-llp-keep", type=float, default=.95)
    parser.add_argument("--min-pt-gev", type=float, default=30.)
    parser.add_argument("--max-abs-eta", type=float, default=1.7)
    parser.add_argument("--skip-hashes", action="store_true", help="Check file sizes only; hashes are verified by default.")
    args = parser.parse_args()
    run_experiment(PROJECT_ROOT, args.output_dir, seed=args.seed,
                   target_llp_keep=args.target_llp_keep, min_pt_GeV=args.min_pt_gev,
                   max_abs_eta=args.max_abs_eta, verify_hashes=not args.skip_hashes)


if __name__ == "__main__":
    main()
