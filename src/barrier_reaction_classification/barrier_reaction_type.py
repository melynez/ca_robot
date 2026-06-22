import argparse
import os
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CONFIG
# =========================================================

CA_ORIGINAL_DIR = None
CA_BARRIER_DIR = None
RANDOM_BARRIER_DIR = None
SAVE_DIR = None
FIG_DIR = None
TABLE_DIR = None

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RULES = list(range(256))
SOURCES = ["shuffle", "white_noise"]
MODE = "easy"
VARIANTS = [1, 2, 3, 4]

BARRIER_GEN = 50
EPSILON = 1e-6

CLASS_ORDER = ["extinction", "loop", "scrambled", "trajectory change", "shift"]


# =========================================================
# METRIC CALCULATION
# =========================================================

class BQMScoreMinimal:
    def __init__(self, epsilon=1e-6, extinction_flag=1e-10):
        self.epsilon = epsilon
        self.extinction_flag = extinction_flag

    def compute_extinction(self, df_original, df_barrier):
        return 1 if len(df_original) != len(df_barrier) else self.epsilon

    def compute_loop(self, df_barrier, barrier_gen):
        barrier_points = df_barrier.iloc[barrier_gen:][["final_x", "final_y"]].to_numpy()
        total_points = len(barrier_points)

        if total_points <= 1:
            return self.epsilon

        coord_repeats = 0
        coord_pairs = 0
        coord_triplets = 0

        for i in range(1, total_points):
            if np.array_equal(barrier_points[i], barrier_points[i - 1]):
                coord_repeats += 1
            elif i > 1 and np.array_equal(barrier_points[i], barrier_points[i - 2]):
                coord_pairs += 1
            elif i > 2 and np.array_equal(barrier_points[i], barrier_points[i - 3]):
                coord_triplets += 1

        repeat_percentage = coord_repeats / total_points * 100
        pair_percentage = coord_pairs / total_points * 100
        triplet_percentage = coord_triplets / total_points * 100

        if repeat_percentage > 70:
            return 0.75
        elif pair_percentage > 70:
            return 0.5
        elif triplet_percentage > 70:
            return 0.25
        else:
            return self.epsilon

    def calculate_orientation_efficiency(self, x, y):
        if len(x) < 2 or len(y) < 2:
            return self.epsilon

        dx = np.diff(x)
        dy = np.diff(y)
        step_lengths = np.sqrt(dx ** 2 + dy ** 2)
        total_path_length = np.sum(step_lengths)

        overall_dx = x[-1] - x[0]
        overall_dy = y[-1] - y[0]
        theta = np.arctan2(dy, dx)
        overall_angle = np.arctan2(overall_dy, overall_dx)

        orientation_errors = theta - overall_angle
        orientation_efficiency = np.sum(step_lengths * np.cos(orientation_errors)) / max(total_path_length, self.epsilon)
        return orientation_efficiency

    def calculate_goal_direction_similarity(self, barrier_point, barrier_endpoint, original_endpoint):
        bx, by = barrier_point
        bx_end, by_end = barrier_endpoint
        ox_end, oy_end = original_endpoint

        angle_to_barrier = np.arctan2(by_end - by, bx_end - bx)
        angle_to_original = np.arctan2(oy_end - by, ox_end - bx)

        angle_difference = abs(angle_to_barrier - angle_to_original)
        angle_difference = min(angle_difference, 2 * np.pi - angle_difference)

        similarity = 1 - (angle_difference / np.pi)
        return similarity

    def calculate_step_length_ratio(self, df_before, df_after):
        def avg_step_length(df):
            x = df["final_x"].to_numpy()
            y = df["final_y"].to_numpy()

            if len(x) < 2 or len(y) < 2:
                return 0

            dx = np.diff(x)
            dy = np.diff(y)
            lengths = np.sqrt(dx ** 2 + dy ** 2)
            return np.mean(lengths) if len(lengths) > 0 else 0

        avg_before = max(avg_step_length(df_before), self.epsilon)
        avg_after = avg_step_length(df_after)
        return avg_after / avg_before

    def compute_filter_metrics(self, df_original, df_barrier, barrier_gen=50):
        extinction = self.compute_extinction(df_original, df_barrier)
        loop = self.compute_loop(df_barrier, barrier_gen)

        # your generated dfBarrier files do not contain this column, so add it here
        df_barrier = df_barrier.copy()
        
        if "generation" in df_barrier.columns:
            df_barrier["after_barrier"] = df_barrier["generation"] >= barrier_gen
        if "gen" in df_barrier.columns:
            df_barrier["after_barrier"] = df_barrier["gen"] >= barrier_gen
        # df_barrier["after_barrier"] = df_barrier["gen"] >= barrier_gen

        df_before = df_barrier[df_barrier["after_barrier"] == False].reset_index(drop=True)
        df_after = df_barrier[df_barrier["after_barrier"] == True].reset_index(drop=True)

        x_before = df_before["final_x"].to_numpy()
        y_before = df_before["final_y"].to_numpy()
        x_after = df_after["final_x"].to_numpy()
        y_after = df_after["final_y"].to_numpy()

        orientation_before = self.calculate_orientation_efficiency(x_before, y_before)
        orientation_after = self.calculate_orientation_efficiency(x_after, y_after)

        barrier_point = df_original.iloc[barrier_gen][["final_x", "final_y"]].values
        barrier_endpoint = df_barrier.iloc[-1][["final_x", "final_y"]].values
        original_endpoint = df_original.iloc[-1][["final_x", "final_y"]].values

        angle_similarity = self.calculate_goal_direction_similarity(
            barrier_point, barrier_endpoint, original_endpoint
        )

        step_length_ratio = self.calculate_step_length_ratio(df_before, df_after)

        return {
            "extinction": extinction,
            "loop": loop,
            "orientation_before": orientation_before,
            "orientation_after": orientation_after,
            "step_length_ratio": step_length_ratio,
            "angle_similarity": angle_similarity,
        }


def classify_barrier_reaction(metrics):
    extinction = metrics.get("extinction", 0)
    loop = metrics.get("loop", 0)
    orientation_before = metrics.get("orientation_before", 0)
    orientation_after = metrics.get("orientation_after", 0)
    angle_similarity = metrics.get("angle_similarity", 1)

    if extinction == 1:
        return "extinction"

    if loop > 0.1:
        return "loop"

    pt_condition = False

    if abs(orientation_after) > 1e-12:
        if (orientation_before / orientation_after) > 3.5:
            pt_condition = True

    if (orientation_before - orientation_after) > 0.3:
        pt_condition = True

    if pt_condition:
        return "scrambled"

    if angle_similarity < 0.6:
        return "trajectory change"

    return "shift"


# =========================================================
# BUILD LABEL TABLES
# =========================================================

def load_ca_labels():
    scorer = BQMScoreMinimal()
    rows = []

    for rule in RULES:
        original_file = CA_ORIGINAL_DIR / f"dfOriginal_r{rule}.csv"
        barrier_file = CA_BARRIER_DIR / f"dfBarrier_r{rule}.csv"

        if not original_file.exists() or not barrier_file.exists():
            continue

        df_original = pd.read_csv(original_file)
        df_barrier = pd.read_csv(barrier_file)

        metrics = scorer.compute_filter_metrics(df_original, df_barrier, barrier_gen=BARRIER_GEN)
        label = classify_barrier_reaction(metrics)

        rows.append({
            "rule": rule,
            "source": "CA",
            "variant": 0,
            "mode": "CA",
            "barrier_reaction_type": label,
            **metrics
        })

    df = pd.DataFrame(rows).sort_values("rule").reset_index(drop=True)
    df.to_csv(TABLE_DIR / "CA_barrier_reaction_labels.csv", index=False)
    return df


def load_random_labels():
    scorer = BQMScoreMinimal()
    rows = []

    for rule in RULES:
        original_file = CA_ORIGINAL_DIR / f"dfOriginal_r{rule}.csv"
        if not original_file.exists():
            continue

        df_original = pd.read_csv(original_file)

        for source in SOURCES:
            for variant in VARIANTS:
                barrier_file = RANDOM_BARRIER_DIR / f"dfBarrier_r{rule}_{source}_v{variant}.csv"

                if not barrier_file.exists():
                    continue

                df_barrier = pd.read_csv(barrier_file)
                metrics = scorer.compute_filter_metrics(df_original, df_barrier, barrier_gen=BARRIER_GEN)
                label = classify_barrier_reaction(metrics)

                rows.append({
                    "rule": rule,
                    "source": source,
                    "variant": variant,
                    "barrier_reaction_type": label,
                    **metrics
                })

    df = pd.DataFrame(rows).sort_values(["rule", "source", "variant", "mode"]).reset_index(drop=True)
    df.to_csv(TABLE_DIR / "random_barrier_reaction_labels.csv", index=False)
    return df


# =========================================================
# MINIMAL SUMMARIES NEEDED FOR PAPER FIGURES
# =========================================================

def compute_distribution_table(ca_df, random_df):
    rows = []

    # CA: one label per rule
    ca_counts = ca_df["barrier_reaction_type"].value_counts()
    total_ca = len(ca_df)
    for cls in CLASS_ORDER:
        rows.append({
            "group": "CA",
            "barrier_reaction_type": cls,
            "count": int(ca_counts.get(cls, 0)),
            "percent": 100 * ca_counts.get(cls, 0) / total_ca if total_ca else 0
        })

    # random sources: 4 easy-mode trajectories per rule-source pooled
    for source in SOURCES:
        sub = random_df[random_df["source"] == source]
        counts = sub["barrier_reaction_type"].value_counts()
        total = len(sub)

        for cls in CLASS_ORDER:
            rows.append({
                "group": source,
                "barrier_reaction_type": cls,
                "count": int(counts.get(cls, 0)),
                "percent": 100 * counts.get(cls, 0) / total if total else 0
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "barrier_reaction_distribution_table.csv", index=False)
    return out


def dominant_label_per_rule(random_df):
    """
    Necessary only to compare one CA label per rule to one random summary label per rule.
    For each rule and source, collapse the 12 random runs to the most frequent label.
    """
    rows = []

    for source in SOURCES:
        sub_source = random_df[random_df["source"] == source]

        for rule, sub_rule in sub_source.groupby("rule"):
            counts = sub_rule["barrier_reaction_type"].value_counts()
            top_label = counts.index[0]
            top_count = counts.iloc[0]

            rows.append({
                "rule": rule,
                "source": source,
                "dominant_barrier_reaction_type": top_label,
                "dominant_count": int(top_count),
                "dominant_fraction": top_count / len(sub_rule)
            })

    out = pd.DataFrame(rows).sort_values(["source", "rule"]).reset_index(drop=True)
    out.to_csv(TABLE_DIR / "random_dominant_label_per_rule.csv", index=False)
    return out


def build_transition_table(ca_df, dominant_df):
    """
    For each source separately:
    CA class -> dominant random class
    """
    rows = []

    ca_map = dict(zip(ca_df["rule"], ca_df["barrier_reaction_type"]))

    for source in SOURCES:
        sub = dominant_df[dominant_df["source"] == source]

        for _, row in sub.iterrows():
            rule = int(row["rule"])
            ca_label = ca_map.get(rule, None)
            rand_label = row["dominant_barrier_reaction_type"]

            if ca_label is None:
                continue

            rows.append({
                "rule": rule,
                "source": source,
                "CA_label": ca_label,
                "random_label": rand_label
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "CA_to_random_transition_table.csv", index=False)
    return out


def build_jaccard_matrix(ca_df, dominant_df, source):
    """
    Jaccard between CA class sets and dominant random class sets.
    """
    sub = dominant_df[dominant_df["source"] == source]

    ca_clusters = {
        cls: set(ca_df.loc[ca_df["barrier_reaction_type"] == cls, "rule"])
        for cls in CLASS_ORDER
    }

    random_clusters = {
        cls: set(sub.loc[sub["dominant_barrier_reaction_type"] == cls, "rule"])
        for cls in CLASS_ORDER
    }

    matrix = pd.DataFrame(index=CLASS_ORDER, columns=CLASS_ORDER, dtype=float)

    for ca_cls in CLASS_ORDER:
        for rand_cls in CLASS_ORDER:
            a = ca_clusters[ca_cls]
            b = random_clusters[rand_cls]
            union = a.union(b)
            intersection = a.intersection(b)
            score = (len(intersection) / len(union) * 100) if union else 0.0
            matrix.loc[ca_cls, rand_cls] = score

    return matrix


# =========================================================
# PLOTS
# =========================================================


#%%


def plot_distribution_figure(dist_df):
    plt.figure(figsize=(10, 6), dpi=400)

    sns.barplot(
        data=dist_df,
        x="barrier_reaction_type",
        y="percent",
        hue="group",
        order=CLASS_ORDER
    )

    plt.ylabel("Percent of trajectories / rules")
    plt.xlabel("Barrier reaction type")
    plt.title("Distribution of barrier reaction types across CA-derived and random bit source controls")    
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_barrier_reaction_distribution.png", bbox_inches="tight")
    plt.close()


def plot_transition_stacked_bars(trans_df):
    """
    Similar in spirit to your old transition figure:
    for each CA label, show distribution of resulting random labels.
    """
    for source in SOURCES:
        sub = trans_df[trans_df["source"] == source].copy()

        if sub.empty:
            continue

        counts = (
            sub.groupby(["CA_label", "random_label"])
            .size()
            .reset_index(name="count")
        )

        pivot = counts.pivot(index="CA_label", columns="random_label", values="count").fillna(0)
        pivot = pivot.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)

        row_sums = pivot.sum(axis=1).replace(0, np.nan)
        pct = pivot.div(row_sums, axis=0)

        plt.figure(figsize=(10, 6), dpi=400)
        bottom = np.zeros(len(pct))

        for cls in CLASS_ORDER:
            vals = pct[cls].fillna(0).values
            plt.bar(pct.index, vals, bottom=bottom, label=cls)
            bottom += vals

        plt.ylim(0, 1)
        plt.ylabel("Fraction of rules")
        plt.xlabel("CA barrier reaction type")
        plt.title(f"CA to {source} dominant barrier reaction type")
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="barrier reaction type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"figure_CA_to_{source}_stacked_transition.png", bbox_inches="tight")
        plt.close()


def plot_jaccard_heatmaps(ca_df, dominant_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=400)

    for ax, source in zip(axes, SOURCES):
        matrix = build_jaccard_matrix(ca_df, dominant_df, source)

        sns.heatmap(
            matrix.T,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            cbar=True,
            linewidths=1,
            square=True,
            ax=ax
        )

        ax.set_title(f"Jaccard Index: CA vs {source}")
        ax.set_xlabel("CA barrier reaction type")
        ax.set_ylabel(f"{source} dominant type")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_CA_vs_random_jaccard_heatmaps.png", bbox_inches="tight")
    plt.close()


# =========================================================
# SIMPLE TEXT SUMMARY
# =========================================================

def write_results_summary(ca_df, random_df, dominant_df):
    lines = []

    lines.append("Barrier reaction type comparison: CA vs randomized bit sources")
    lines.append("")

    # overall counts
    for label_name, df_in in [
        ("CA", ca_df),
        ("shuffle", random_df[random_df["source"] == "shuffle"]),
        ("white_noise", random_df[random_df["source"] == "white_noise"]),
    ]:
        counts = df_in["barrier_reaction_type"].value_counts(normalize=True) * 100
        lines.append(f"{label_name} distribution (%):")
        for cls in CLASS_ORDER:
            lines.append(f"  {cls}: {counts.get(cls, 0):.1f}")
        lines.append("")

    # CA vs dominant random agreement
    ca_map = dict(zip(ca_df["rule"], ca_df["barrier_reaction_type"]))
    for source in SOURCES:
        sub = dominant_df[dominant_df["source"] == source]
        matches = []

        for _, row in sub.iterrows():
            rule = int(row["rule"])
            ca_label = ca_map.get(rule, None)
            rand_label = row["dominant_barrier_reaction_type"]
            if ca_label is not None:
                matches.append(ca_label == rand_label)

        match_rate = np.mean(matches) * 100 if matches else np.nan
        lines.append(f"CA vs {source} dominant-label agreement: {match_rate:.1f}%")

    lines.append("")

    # most unstable rules in random dominance
    lines.append("Note:")
    lines.append(
    "The dominant random label per rule is the most frequent label among the 4 easy-mode random variants "
    "for that rule and source."
    )

    out_file = TABLE_DIR / "results_summary.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CA, shuffle, and white-noise barrier reaction types."
    )

    parser.add_argument("--ca_original_dir", required=True)
    parser.add_argument("--ca_barrier_dir", required=True)
    parser.add_argument("--random_barrier_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--barrier_gen", type=int, default=50)
    parser.add_argument("--rules_start", type=int, default=0)
    parser.add_argument("--rules_end", type=int, default=255)

    parser.add_argument("--sources", nargs="+", default=["shuffle", "white_noise"],
                        choices=["shuffle", "white_noise"])
    parser.add_argument("--variants", type=int, default=4)

    return parser.parse_args()

def main():
    global CA_ORIGINAL_DIR, CA_BARRIER_DIR, RANDOM_BARRIER_DIR
    global SAVE_DIR, FIG_DIR, TABLE_DIR
    global RULES, SOURCES, MODES, VARIANTS, BARRIER_GEN

    args = parse_args()

    CA_ORIGINAL_DIR = Path(args.ca_original_dir)
    CA_BARRIER_DIR = Path(args.ca_barrier_dir)
    RANDOM_BARRIER_DIR = Path(args.random_barrier_dir)

    SAVE_DIR = Path(args.out_dir)
    FIG_DIR = SAVE_DIR / "figures"
    TABLE_DIR = SAVE_DIR / "tables"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    RULES = list(range(args.rules_start, args.rules_end + 1))
    SOURCES = args.sources
    MODES = args.modes
    VARIANTS = list(range(1, args.variants + 1))
    BARRIER_GEN = args.barrier_gen

    print("\n===== BARRIER REACTION TYPE COMPARISON =====")
    print(f"CA original dir    : {CA_ORIGINAL_DIR}")
    print(f"CA barrier dir     : {CA_BARRIER_DIR}")
    print(f"Random barrier dir : {RANDOM_BARRIER_DIR}")
    print(f"Barrier mode       : easy")
    print(f"Sources            : {SOURCES}")
    print(f"Variants/source    : {len(VARIANTS)}")
    print(f"Rules              : {RULES[0]}-{RULES[-1]}")
    print("============================================\n")
    
    print("Computing CA barrier reaction labels...")
    ca_df = load_ca_labels()

    print("Computing random barrier reaction labels...")
    random_df = load_random_labels()

    print("Building essential summary tables...")
    dist_df = compute_distribution_table(ca_df, random_df)
    dominant_df = dominant_label_per_rule(random_df)
    trans_df = build_transition_table(ca_df, dominant_df)

    print("Making figures...")
    plot_distribution_figure(dist_df)
    plot_transition_stacked_bars(trans_df)
    plot_jaccard_heatmaps(ca_df, dominant_df)

    print("Writing text summary...")
    write_results_summary(ca_df, random_df, dominant_df)

    print("Done.")
    print(f"Tables saved to: {TABLE_DIR}")
    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
    
    