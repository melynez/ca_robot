import argparse
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import gymnasium as gym
import minigrid


LEFT = 0
RIGHT = 1
FORWARD = 2

BIT_TO_ACTION = {
    "00": LEFT,
    "01": RIGHT,
    "10": FORWARD,
    "11": FORWARD,
}

ACTION_NAME = {
    LEFT: "left",
    RIGHT: "right",
    FORWARD: "forward",
}


# =========================
# SOURCES
# =========================

def rule_to_dict(rule_num: int) -> Dict[str, int]:
    s = format(rule_num, "08b")
    return {format(i, "03b"): int(s[7 - i]) for i in range(8)}


class CASource:
    def __init__(self, rule_num: int, initial_state: str = "0001000"):
        self.rule_num = rule_num
        self.rule = rule_to_dict(rule_num)
        self.current_state = initial_state
        self.bg_type = "0"
        self.generation_index = -1

    def next_generation(self) -> str:
        if self.generation_index == -1:
            self.generation_index = 0
            return self.current_state

        expanded = self.bg_type * 2 + self.current_state + self.bg_type * 2
        new_state = "".join(
            str(self.rule[expanded[i:i + 3]])
            for i in range(len(expanded) - 2)
        )

        self.bg_type = str(self.rule[self.bg_type * 3])
        self.current_state = new_state
        self.generation_index += 1
        return new_state


class ShuffleSource:
    def __init__(self, rule_num: int, initial_state: str = "0001000", seed: int = 0):
        self.ca = CASource(rule_num, initial_state)
        self.rng = random.Random(seed)

    def next_generation(self) -> str:
        chars = list(self.ca.next_generation())
        self.rng.shuffle(chars)
        return "".join(chars)


class WhiteNoiseSource:
    def __init__(self, initial_length: int = 7, seed: int = 0):
        self.length = initial_length
        self.rng = random.Random(seed)

    def next_generation(self) -> str:
        if self.length % 2 == 0:
            zeros = ones = self.length // 2
        else:
            zeros = self.length // 2
            ones = self.length // 2 + 1

        chars = ["0"] * zeros + ["1"] * ones
        self.rng.shuffle(chars)
        out = "".join(chars)
        self.length += 2
        return out


# =========================
# MINIGRID HELPERS
# =========================

def front_position(agent_pos: Tuple[int, int], agent_dir: int) -> Tuple[int, int]:
    x, y = agent_pos

    if agent_dir == 0:      # facing right
        return x + 1, y
    elif agent_dir == 1:    # facing down
        return x, y + 1
    elif agent_dir == 2:    # facing left
        return x - 1, y
    else:                   # facing up
        return x, y - 1


def forward_blocked(env) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    u = env.unwrapped
    fx, fy = front_position(tuple(u.agent_pos), int(u.agent_dir))
    cell = u.grid.get(fx, fy)

    if cell is None:
        return False, None, (fx, fy)

    if getattr(cell, "type", None) == "wall":
        return True, "wall", (fx, fy)

    can_overlap = getattr(cell, "can_overlap", None)
    if callable(can_overlap) and not can_overlap():
        return True, getattr(cell, "type", "blocked"), (fx, fy)

    return False, None, (fx, fy)


def goal_position(env):
    u = env.unwrapped
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                return x, y
    return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =========================
# EXACT PER-STEP RETRY LOGIC
# =========================

def run_one_episode_exact_retries(
    source_label: str,
    source,
    env_name: str = "MiniGrid-FourRooms-v0",
    seed: int = 0,
    num_generations: int = 100,
    env_max_steps: int = 100000,
):
    env = gym.make(env_name)

    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = env_max_steps

    obs, info = env.reset(seed=seed)

    attempt_rows = []
    step_rows = []
    generation_rows = []

    accepted_step_global = 0
    success = False
    truncated = False

    goal = goal_position(env)

    for generation_index in range(num_generations):
        original_bits = source.next_generation()
        active_bits = list(original_bits)

        i = 0
        retries_waiting_for_next_accept = 0
        attempts_this_generation = 0
        rejections_this_generation = 0
        accepted_this_generation = 0

        while len(active_bits) >= 2 and i < len(active_bits):
            i1 = i
            i2 = 0 if i1 == len(active_bits) - 1 else i1 + 1
            chunk = active_bits[i1] + active_bits[i2]
            action = BIT_TO_ACTION[chunk]

            u = env.unwrapped
            x_before, y_before = int(u.agent_pos[0]), int(u.agent_pos[1])
            dir_before = int(u.agent_dir)

            old_dist = None
            if goal is not None:
                old_dist = manhattan((x_before, y_before), goal)

            blocked = False
            rejection_reason = None
            front_x, front_y = front_position((x_before, y_before), dir_before)

            if action == FORWARD:
                blocked, rejection_reason, (front_x, front_y) = forward_blocked(env)

            attempts_this_generation += 1

            attempt_row = {
                "source_label": source_label,
                "seed": seed,
                "generation_index": generation_index,
                "attempt_index_in_generation": attempts_this_generation,
                "accepted_step_global_if_accepts": accepted_step_global,
                "active_bits_before": "".join(active_bits),
                "original_generation_bits": original_bits,
                "generation_length_before": len(original_bits),
                "active_length_before_attempt": len(active_bits),
                "pair_first_index": i1,
                "pair_second_index": i2,
                "chunk": chunk,
                "decoded_action": action,
                "decoded_action_name": ACTION_NAME[action],
                "agent_x_before": x_before,
                "agent_y_before": y_before,
                "agent_dir_before": dir_before,
                "front_x": front_x,
                "front_y": front_y,
                "valid": not blocked,
                "rejected_by_wrapper": blocked,
                "rejection_reason": rejection_reason,
                "retries_waiting_before_this_attempt": retries_waiting_for_next_accept,
            }

            if blocked:
                removed_bit = active_bits.pop(i2)
                rejections_this_generation += 1
                retries_waiting_for_next_accept += 1

                attempt_row.update({
                    "removed_bit": removed_bit,
                    "removed_index": i2,
                    "active_bits_after": "".join(active_bits),
                    "accepted_step_global": None,
                    "retries_assigned_to_accepted_step": None,
                })

                attempt_rows.append(attempt_row)

                # retry from same local location after bit removal
                i = min(i1, len(active_bits))
                continue

            # Accepted action
            obs, reward, terminated, step_truncated, info = env.step(action)

            u2 = env.unwrapped
            x_after, y_after = int(u2.agent_pos[0]), int(u2.agent_pos[1])
            dir_after = int(u2.agent_dir)

            new_dist = None
            delta_manhattan = None
            if goal is not None and old_dist is not None:
                new_dist = manhattan((x_after, y_after), goal)
                delta_manhattan = old_dist - new_dist

            attempt_row.update({
                "removed_bit": None,
                "removed_index": None,
                "active_bits_after": "".join(active_bits),
                "accepted_step_global": accepted_step_global,
                "retries_assigned_to_accepted_step": retries_waiting_for_next_accept,
            })
            attempt_rows.append(attempt_row)

            step_rows.append({
                "source_label": source_label,
                "seed": seed,
                "generation_index": generation_index,
                "accepted_step_global": accepted_step_global,
                "step_index_in_generation": accepted_this_generation,
                "chunk_used": chunk,
                "accepted_action": action,
                "accepted_action_name": ACTION_NAME[action],
                "agent_x_before": x_before,
                "agent_y_before": y_before,
                "agent_dir_before": dir_before,
                "agent_x_after": x_after,
                "agent_y_after": y_after,
                "agent_dir_after": dir_after,
                "goal_x": None if goal is None else goal[0],
                "goal_y": None if goal is None else goal[1],
                "manhattan_before": old_dist,
                "manhattan_after": new_dist,
                "delta_manhattan": delta_manhattan,

                # THIS IS THE EXACT SURPRISE VALUE
                "retries_before_accept": retries_waiting_for_next_accept,
                "attempts_before_accept": retries_waiting_for_next_accept + 1,

                "reward": float(reward),
                "terminated_here": bool(terminated),
                "truncated_here": bool(step_truncated),
            })

            accepted_step_global += 1
            accepted_this_generation += 1
            retries_waiting_for_next_accept = 0
            i += 1

            if terminated:
                success = True
                break

            if step_truncated:
                truncated = True
                break

        removed_bits_generation = len(original_bits) - len(active_bits)

        generation_rows.append({
            "source_label": source_label,
            "seed": seed,
            "generation_index": generation_index,
            "generation_length_before": len(original_bits),
            "generation_length_after": len(active_bits),
            "generation_removed_bits": removed_bits_generation,
            "generation_brr": (
                removed_bits_generation / len(original_bits)
                if len(original_bits) > 0 else 0
            ),
            "attempts_in_generation": attempts_this_generation,
            "rejections_in_generation": rejections_this_generation,
            "accepted_steps_in_generation": accepted_this_generation,
            "dangling_retries_end_generation": retries_waiting_for_next_accept,
        })

        if success or truncated:
            break

    env.close()

    attempt_df = pd.DataFrame(attempt_rows)
    step_df = pd.DataFrame(step_rows)
    generation_df = pd.DataFrame(generation_rows)

    total_generated_bits = generation_df["generation_length_before"].sum()
    total_removed_bits = generation_df["generation_removed_bits"].sum()
    total_attempts = len(attempt_df)
    total_rejections = int(attempt_df["rejected_by_wrapper"].sum()) if not attempt_df.empty else 0
    total_accepted_steps = len(step_df)

    summary_df = pd.DataFrame([{
        "source_label": source_label,
        "seed": seed,
        "success": success,
        "truncated": truncated,
        "generations_consumed": len(generation_df),
        "accepted_env_steps_total": total_accepted_steps,
        "total_attempts": total_attempts,
        "total_rejections": total_rejections,
        "total_generated_bits": total_generated_bits,
        "total_removed_bits": total_removed_bits,
        "brr_overall": (
            total_removed_bits / total_generated_bits
            if total_generated_bits > 0 else 0
        ),

        # exact surprise summary
        "mean_retries_per_accepted_step": (
            step_df["retries_before_accept"].mean()
            if total_accepted_steps > 0 else None
        ),
        "median_retries_per_accepted_step": (
            step_df["retries_before_accept"].median()
            if total_accepted_steps > 0 else None
        ),
        "p90_retries_per_accepted_step": (
            step_df["retries_before_accept"].quantile(0.90)
            if total_accepted_steps > 0 else None
        ),
        "max_retries_per_accepted_step": (
            step_df["retries_before_accept"].max()
            if total_accepted_steps > 0 else None
        ),
    }])

    return attempt_df, step_df, generation_df, summary_df


# =========================
# BATCH RUN
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MiniGrid CAR, shuffle, and white-noise agents with exact retry logging."
    )

    parser.add_argument("--out_dir", default="outputs/minigrid_exact_retry_outputs")
    parser.add_argument("--env_name", default="MiniGrid-FourRooms-v0")

    parser.add_argument("--initial_state", default="0001000")
    parser.add_argument("--rules_start", type=int, default=0)
    parser.add_argument("--rules_end", type=int, default=255)
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds to run, using seeds 0 to seeds-1")
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--env_max_steps", type=int, default=100000)

    parser.add_argument("--sources", nargs="+",
                        default=["ca", "shuffle", "white_noise"],
                        choices=["ca", "shuffle", "white_noise"])

    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INITIAL_STATE = args.initial_state
    RULES = list(range(args.rules_start, args.rules_end + 1))
    SEEDS = list(range(args.seeds))
    NUM_GENERATIONS = args.num_generations

    print("\n===== MINIGRID EXACT RETRY RUN =====")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Environment     : {args.env_name}")
    print(f"Initial state   : {INITIAL_STATE}")
    print(f"Rules           : {args.rules_start}-{args.rules_end}")
    print(f"Seeds           : 0-{args.seeds - 1}")
    print(f"Generations     : {NUM_GENERATIONS}")
    print(f"Sources         : {args.sources}")
    print("====================================\n")

    all_attempts = []
    all_steps = []
    all_generations = []
    all_summaries = []

    # white noise global baseline
    if "white_noise" in args.sources:
        for seed in SEEDS:
            att, steps, gens, summary = run_one_episode_exact_retries(
                source_label="white_noise_v1",
                source=WhiteNoiseSource(initial_length=len(INITIAL_STATE), seed=seed),
                env_name=args.env_name,
                seed=seed,
                num_generations=NUM_GENERATIONS,
                env_max_steps=args.env_max_steps,
            )
            all_attempts.append(att)
            all_steps.append(steps)
            all_generations.append(gens)
            all_summaries.append(summary)

    # CA
    if "ca" in args.sources:
        for rule in RULES:
            for seed in SEEDS:
                att, steps, gens, summary = run_one_episode_exact_retries(
                    source_label=f"ca_rule{rule}",
                    source=CASource(rule_num=rule, initial_state=INITIAL_STATE),
                    env_name=args.env_name,
                    seed=seed,
                    num_generations=NUM_GENERATIONS,
                    env_max_steps=args.env_max_steps,
                )
                all_attempts.append(att)
                all_steps.append(steps)
                all_generations.append(gens)
                all_summaries.append(summary)

    # shuffle
    if "shuffle" in args.sources:
        for rule in RULES:
            for seed in SEEDS:
                att, steps, gens, summary = run_one_episode_exact_retries(
                    source_label=f"shuffle_rule{rule}",
                    source=ShuffleSource(rule_num=rule, initial_state=INITIAL_STATE, seed=seed),
                    env_name=args.env_name,
                    seed=seed,
                    num_generations=NUM_GENERATIONS,
                    env_max_steps=args.env_max_steps,
                )
                all_attempts.append(att)
                all_steps.append(steps)
                all_generations.append(gens)
                all_summaries.append(summary)

    if not all_summaries:
        raise RuntimeError("No runs were completed. Check --sources, --rules, and --seeds.")

    pd.concat(all_attempts, ignore_index=True).to_csv(
        OUTPUT_DIR / "attempt_log_exact.csv", index=False
    )

    pd.concat(all_steps, ignore_index=True).to_csv(
        OUTPUT_DIR / "step_metrics_exact_retries.csv", index=False
    )

    pd.concat(all_generations, ignore_index=True).to_csv(
        OUTPUT_DIR / "generation_metrics_exact.csv", index=False
    )

    pd.concat(all_summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "run_summary_exact_retries.csv", index=False
    )

    print("Saved exact retry outputs to:", OUTPUT_DIR.resolve())
    print(f"Total runs completed: {len(all_summaries)}")


if __name__ == "__main__":
    main()