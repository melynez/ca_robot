"""
generate_car_control_path.py

Creates shuffle and white-noise control trajectories
from CA-generated reference trajectories.

Inputs
------
dfOriginal_r{rule}.csv

Outputs
-------
dfOriginal_r{rule}_{source}_v{variant}.csv
dfBarrier_r{rule}_{source}_v{variant}_{mode}.csv
dicBarrier_r{rule}_{source}_v{variant}_{mode}.csv
"""

import argparse
import ast
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

def rule_to_string(rule_num):
    if rule_num < 0 or rule_num > 255:
        raise ValueError("Rule number must be between 0 and 255")
    strRule = format(rule_num, '08b')
    dicRule = {format(i, '03b'): int(strRule[7 - i]) for i in range(8)}
    return dicRule

# Function to simulate robot movement without barriers using generation-based iteration
def dir_original(
    df_reference,
    source_type,
    variant_num,
    rng=None,
    initial_state='0001000',
    initial_position=(0, 0)
):    
    """
    Build ONE original (no-barrier) random-control dataframe for ONE run:
    - source_type: 'shuffle' or 'white_noise'
    - variant_num: 1,2,3,4

    Returns:
        dfOriginal
    """

    if rng is None:
        rng = random.Random()

    x_init, y_init = initial_position
    rows = []

    # generation 0 fixed
    rows.append({
        'generation': 0,
        'bin_string': initial_state,
        'final_x': float(x_init),
        'final_y': float(y_init)
    })

    prev_x, prev_y = float(x_init), float(y_init)

    for gen in range(1, len(df_reference)):
        reference_string = str(df_reference.iloc[gen]['bin_string'])

        if source_type == 'shuffle':
            candidate_string = apply_shuffle(
                reference_string=reference_string,
                rng=rng
            )

        elif source_type == 'white_noise':
            candidate_string = apply_white_noise(
                reference_string=reference_string,
                variant_num=variant_num,
                rng=rng
            )

        else:
            raise ValueError("source_type must be 'shuffle' or 'white_noise'")

        final_x, final_y = find_xy(candidate_string, prev_x, prev_y)

        rows.append({
            'generation': gen,
            'bin_string': candidate_string,
            'final_x': final_x,
            'final_y': final_y
        })

        prev_x, prev_y = final_x, final_y

    dfOriginal = pd.DataFrame(rows)
    return dfOriginal



# ============= dfBarrier ==================

def cross_test(p1, p2, p3, p4, epsilon=0.009):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(den) < epsilon:
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    return (-epsilon <= t <= 1 + epsilon) and (-epsilon <= u <= 1 + epsilon)

def calculate_barrier(barrier_point, slope, barrier_extend, angle_degrees):
    bx, by = barrier_point 
    angle_radians = math.radians(angle_degrees)

    if slope is None: # vertical barrier
        barrier_start = (bx, by - barrier_extend)
        barrier_end = (bx, by + barrier_extend)
    elif slope == 0: # horizontal barrier
        barrier_start = (bx - barrier_extend, by)
        barrier_end = (bx + barrier_extend, by)
    else: 
        barrier_slope = math.tan(angle_radians + math.atan(slope))

        dx = barrier_extend / math.sqrt(1 + barrier_slope**2)
        dy = barrier_slope * dx

        barrier_start = (bx - dx, by - dy)
        barrier_end = (bx + dx, by + dy)
    return barrier_start, barrier_end


def find_xy(bitstring, prev_x, prev_y):
    dicDirection = {
        '00': (0, 1),
        '01': (1, 0),
        '10': (0, -1),
        '11': (-1, 0)
    }

    temp_x, temp_y = prev_x, prev_y

    if len(bitstring) <= 1:
        return temp_x, temp_y

    for i in range(len(bitstring)):
        if i == len(bitstring) - 1:
            two_bit = bitstring[i] + bitstring[0]
        else:
            two_bit = bitstring[i:i + 2]

        dx, dy = dicDirection.get(two_bit, (0, 0))
        temp_x += dx
        temp_y += dy

    return temp_x, temp_y

def load_barrier_info(dicbarrier_csv_path):
    df = pd.read_csv(dicbarrier_csv_path)
    row = df.iloc[0].to_dict()

    barrier_start = row.get('barrier_start')
    barrier_end = row.get('barrier_end')

    if isinstance(barrier_start, str):
        barrier_start = ast.literal_eval(barrier_start)
    if isinstance(barrier_end, str):
        barrier_end = ast.literal_eval(barrier_end)

    return {
        'barrier_start': barrier_start,
        'barrier_end': barrier_end,
        'barrier_extend': row.get('barrier_extend'),
        'barrier_angle': row.get('barrier_angle')
    }
    
    
def count_bits(bitstring):
    return (
        len(bitstring),
        bitstring.count('0'),
        bitstring.count('1')
    )


# =================== BITSOURCE TYPE ==================

def apply_shuffle(reference_string, rng=None, max_tries=1000):
    if rng is None:
        rng = random.Random()

    chars = list(reference_string)

    if len(set(chars)) == 1:
        return reference_string

    for _ in range(max_tries):
        shuffled = chars[:]
        rng.shuffle(shuffled)
        candidate = ''.join(shuffled)

        if candidate != reference_string:
            return candidate

    return reference_string


def apply_white_noise(reference_string, variant_num, rng=None, max_tries=1000):
    if rng is None:
        rng = random.Random()

    length = len(reference_string)

    if length == 0:
        return ''

    if length % 2 == 0:
        num_0 = length // 2
        num_1 = length // 2
    else:
        if variant_num in [1, 2]:
            num_1 = (length // 2) + 1
            num_0 = length // 2
        elif variant_num in [3, 4]:
            num_0 = (length // 2) + 1
            num_1 = length // 2
        else:
            raise ValueError("variant_num must be 1, 2, 3, or 4")

    chars = ['0'] * num_0 + ['1'] * num_1

    # only one possible arrangement
    if len(set(chars)) == 1:
        return ''.join(chars)

    for _ in range(max_tries):
        trial = chars[:]
        rng.shuffle(trial)
        candidate = ''.join(trial)

        if candidate != reference_string:
            return candidate

    # fallback
    return ''.join(chars)

# =================== BARRIER_MODE ==================

def remove_crossing_bits_once(candidate_string, prev_x, prev_y, barrier_start, barrier_end):
    dicDirection = {
        '00': (0, 1),
        '01': (1, 0),
        '10': (0, -1),
        '11': (-1, 0)
    }

    length_before = len(candidate_string)
    num_0_before = candidate_string.count('0')
    num_1_before = candidate_string.count('1')

    candidate_x, candidate_y = find_xy(candidate_string, prev_x, prev_y)
    does_cross = cross_test((prev_x, prev_y), (candidate_x, candidate_y), barrier_start, barrier_end)

    if not does_cross:
        return {
            'bin_string': candidate_string,
            'final_x': candidate_x,
            'final_y': candidate_y,
            'crosses_barrier': False,
            'length_before': length_before,
            'length_after': length_before,
            'bits_removed': 0,
            'num_0_before': num_0_before,
            'num_1_before': num_1_before,
            'num_0_after': num_0_before,
            'num_1_after': num_1_before,
            'extinction': False        
            }

    if len(candidate_string) <= 1:
        return {
            'bin_string': '',
            'final_x': prev_x,
            'final_y': prev_y,
            'crosses_barrier': True,
            'length_before': length_before,
            'length_after': 0,
            'bits_removed': length_before,
            'num_0_before': num_0_before,
            'num_1_before': num_1_before,
            'num_0_after': 0,
            'num_1_after': 0,
            'extinction': True
        }

    flagged_bits = []
    current_array = []
    i1 = 0
    num_cross = 0
    n = len(candidate_string)

    while i1 <= n - 1:
        i2 = i1 + 1 + num_cross

        if i2 == n:
            i2 = 0
        elif i2 > n:
            break

        bit1 = candidate_string[i1]
        bit2 = candidate_string[i2]
        two_bit = bit1 + bit2
        dx, dy = dicDirection.get(two_bit, (0, 0))

        if len(current_array) == 0:
            temp_x = prev_x + dx
            temp_y = prev_y + dy
        else:
            temp_x = current_array[-1][3] + dx
            temp_y = current_array[-1][4] + dy

        if cross_test((prev_x, prev_y), (temp_x, temp_y), barrier_start, barrier_end):
            flagged_bits.append(i2)
            num_cross += 1
        else:
            current_array.append((two_bit, dx, dy, temp_x, temp_y))
            i1 += 1 + num_cross
            num_cross = 0

    final_string = ''.join(bit for idx, bit in enumerate(candidate_string) if idx not in flagged_bits)

    length_after = len(final_string)
    num_0_after = final_string.count('0')
    num_1_after = final_string.count('1')
    bits_removed = length_before - length_after

    if length_after <= 1 or len(current_array) == 0:
        return {
            'bin_string': '',
            'final_x': prev_x,
            'final_y': prev_y,
            'crosses_barrier': True,
            'length_before': length_before,
            'length_after': 0,
            'bits_removed': length_before,
            'num_0_before': num_0_before,
            'num_1_before': num_1_before,
            'num_0_after': 0,
            'num_1_after': 0,
            'extinction': True
        }

    final_x = current_array[-1][3]
    final_y = current_array[-1][4]

    return {
        'bin_string': final_string,
        'final_x': final_x,
        'final_y': final_y,
        'crosses_barrier': True,
        'length_before': length_before,
        'length_after': length_after,
        'bits_removed': bits_removed,
        'num_0_before': num_0_before,
        'num_1_before': num_1_before,
        'num_0_after': num_0_after,
        'num_1_after': num_1_after,
        'extinction': False
    }
    
    
def hard_mode_step(candidate_string, prev_x, prev_y, barrier_start, barrier_end):
    length_before = len(candidate_string)
    num_0_before = candidate_string.count('0')
    num_1_before = candidate_string.count('1')

    candidate_x, candidate_y = find_xy(candidate_string, prev_x, prev_y)
    does_cross = cross_test((prev_x, prev_y), (candidate_x, candidate_y), barrier_start, barrier_end)

    if does_cross:
        return {
            'bin_string': '',
            'final_x': prev_x,
            'final_y': prev_y,
            'crosses_barrier': True,
            'attempts_used': 1,
            'extinction': True,
            'length_before': length_before,
            'length_after': 0,
            'bits_removed': length_before,
            'num_0_before': num_0_before,
            'num_1_before': num_1_before,
            'num_0_after': 0,
            'num_1_after': 0
        }

    return {
        'bin_string': candidate_string,
        'final_x': candidate_x,
        'final_y': candidate_y,
        'crosses_barrier': False,
        'attempts_used': 1,
        'extinction': False,
        'length_before': length_before,
        'length_after': length_before,
        'bits_removed': 0,
        'num_0_before': num_0_before,
        'num_1_before': num_1_before,
        'num_0_after': num_0_before,
        'num_1_after': num_1_before
    }

def easy_mode_step(candidate_string, prev_x, prev_y, barrier_start, barrier_end):
    attempts = 0
    current_string = candidate_string
    first_cross = False
    first_before_len = len(candidate_string)
    first_num_0 = candidate_string.count('0')
    first_num_1 = candidate_string.count('1')

    while True:
        attempts += 1
        result = remove_crossing_bits_once(current_string, prev_x, prev_y, barrier_start, barrier_end)

        if attempts == 1:
            first_cross = result['crosses_barrier']

        if not result['crosses_barrier']:
            result['crosses_barrier'] = first_cross
            result['attempts_used'] = attempts
            result['length_before'] = first_before_len
            result['num_0_before'] = first_num_0
            result['num_1_before'] = first_num_1
            result['bits_removed'] = first_before_len - result['length_after']
            return result

        if result['extinction']:
            result['crosses_barrier'] = first_cross
            result['attempts_used'] = attempts
            result['length_before'] = first_before_len
            result['num_0_before'] = first_num_0
            result['num_1_before'] = first_num_1
            result['bits_removed'] = first_before_len
            return result

        current_string = result['bin_string']
        
def medium_mode_step(candidate_string, prev_x, prev_y, barrier_start, barrier_end, K=3):
    attempts = 0
    current_string = candidate_string
    first_cross = False
    first_before_len = len(candidate_string)
    first_num_0 = candidate_string.count('0')
    first_num_1 = candidate_string.count('1')

    while True:
        attempts += 1
        result = remove_crossing_bits_once(current_string, prev_x, prev_y, barrier_start, barrier_end)

        if attempts == 1:
            first_cross = result['crosses_barrier']

        if not result['crosses_barrier']:
            result['crosses_barrier'] = first_cross
            result['attempts_used'] = attempts
            result['length_before'] = first_before_len
            result['num_0_before'] = first_num_0
            result['num_1_before'] = first_num_1
            result['bits_removed'] = first_before_len - result['length_after']
            return result

        if result['extinction']:
            result['crosses_barrier'] = first_cross
            result['attempts_used'] = attempts
            result['length_before'] = first_before_len
            result['num_0_before'] = first_num_0
            result['num_1_before'] = first_num_1
            result['bits_removed'] = first_before_len
            return result

        if attempts >= K:
            return {
                'bin_string': '',
                'final_x': prev_x,
                'final_y': prev_y,
                'crosses_barrier': first_cross,
                'attempts_used': attempts,
                'extinction': True,
                'length_before': first_before_len,
                'length_after': 0,
                'bits_removed': first_before_len,
                'num_0_before': first_num_0,
                'num_1_before': first_num_1,
                'num_0_after': 0,
                'num_1_after': 0
            }

        current_string = result['bin_string']
# ====================== DIR_BARRIER FUNCTION =========================

def dir_barrier(
    df_reference,
    df_random_original,
    source_type,
    variant_num,
    barrier_gen,
    barrier_extend,
    barrier_angle,
    barrier_mode,
    rng=None,
    K=3,
    initial_state='0001000',
    initial_position=(0, 0)
):
    """
    Build ONE barrier trajectory dataframe for ONE run.

    Intended behavior:
    - df_random_original is the source trajectory for this run
    - pre-barrier rows in dfBarrier exactly match df_random_original
    - dicBarrier is defined from df_random_original
    - after the barrier starts, candidate_string also comes from df_random_original
    - barrier logic is then applied to those candidate strings
    """

    if rng is None:
        rng = random.Random()

    rows = []
    extinction = False

    dicBarrier = {
        'barrier_extend': barrier_extend,
        'barrier_angle': barrier_angle
    }

    n_rows = len(df_random_original)
    if n_rows == 0:
        return pd.DataFrame(rows), dicBarrier, False

    # -----------------------------------------
    # old offset convention:
    # if barrier_gen = 49, first barrier-tested generation is gen = 48
    # so pre-barrier rows are gen < barrier_gen - 1
    # -----------------------------------------
    barrier_start_gen = barrier_gen - 1

    # -----------------------------------------
    # 1) copy pre-barrier rows directly from df_random_original
    # -----------------------------------------
    for gen in range(0, min(barrier_start_gen, n_rows)):
        s = str(df_random_original.iloc[gen]['bin_string'])

        rows.append({
            'generation': int(df_random_original.iloc[gen]['generation']),
            'bin_string': s,
            'final_x': float(df_random_original.iloc[gen]['final_x']),
            'final_y': float(df_random_original.iloc[gen]['final_y']),
            'crosses_barrier': False,
            'attempts_used': 1,
            'extinction': False,
            'length_before': len(s),
            'length_after': len(s),
            'bits_removed': 0,
            'num_0_before': s.count('0'),
            'num_1_before': s.count('1'),
            'num_0_after': s.count('0'),
            'num_1_after': s.count('1')
        })

    # If barrier would start before generation 1, clamp safely
    if barrier_start_gen < 0:
        barrier_start_gen = 0

    # If barrier start is beyond available rows, just return copied original
    if barrier_start_gen >= n_rows:
        dfBarrier = pd.DataFrame(rows)
        return dfBarrier, dicBarrier, False

    # -----------------------------------------
    # 2) define barrier from df_random_original
    # -----------------------------------------
    # keep your old-style convention:
    # anchor at row barrier_gen when possible
    anchor_idx = min(barrier_gen, n_rows - 1)
    slope_prev_idx = max(0, min(barrier_gen - 1, n_rows - 1))
    slope_curr_idx = min(barrier_gen, n_rows - 1)

    barrier_point = (
        float(df_random_original.iloc[anchor_idx]['final_x']),
        float(df_random_original.iloc[anchor_idx]['final_y'])
    )

    prev_fullx = float(df_random_original.iloc[slope_prev_idx]['final_x'])
    prev_fully = float(df_random_original.iloc[slope_prev_idx]['final_y'])
    temp_fullx = float(df_random_original.iloc[slope_curr_idx]['final_x'])
    temp_fully = float(df_random_original.iloc[slope_curr_idx]['final_y'])

    if prev_fullx != temp_fullx:
        ca_slope = (prev_fully - temp_fully) / (prev_fullx - temp_fullx)
    else:
        ca_slope = 0

    barrier_start, barrier_end = calculate_barrier(
        barrier_point=barrier_point,
        slope=ca_slope,
        barrier_extend=barrier_extend,
        angle_degrees=barrier_angle
    )

    dicBarrier['barrier_gen'] = barrier_gen
    dicBarrier['barrier_start'] = barrier_start
    dicBarrier['barrier_end'] = barrier_end
    dicBarrier['barrier_mode'] = barrier_mode
    if barrier_mode == 'medium':
        dicBarrier['K'] = K

    # -----------------------------------------
    # 3) start barrier trajectory from the copied original point
    # -----------------------------------------
    if barrier_start_gen == 0:
        prev_x, prev_y = float(initial_position[0]), float(initial_position[1])
    else:
        prev_x = float(df_random_original.iloc[barrier_start_gen - 1]['final_x'])
        prev_y = float(df_random_original.iloc[barrier_start_gen - 1]['final_y'])

    # -----------------------------------------
    # 4) apply barrier logic using candidate strings
    #    from df_random_original itself
    # -----------------------------------------
    for gen in range(barrier_start_gen, n_rows):
        candidate_string = str(df_random_original.iloc[gen]['bin_string'])

        if barrier_mode == 'hard':
            row = hard_mode_step(
                candidate_string=candidate_string,
                prev_x=prev_x,
                prev_y=prev_y,
                barrier_start=barrier_start,
                barrier_end=barrier_end
            )

        elif barrier_mode == 'easy':
            row = easy_mode_step(
                candidate_string=candidate_string,
                prev_x=prev_x,
                prev_y=prev_y,
                barrier_start=barrier_start,
                barrier_end=barrier_end
            )

        elif barrier_mode == 'medium':
            row = medium_mode_step(
                candidate_string=candidate_string,
                prev_x=prev_x,
                prev_y=prev_y,
                barrier_start=barrier_start,
                barrier_end=barrier_end,
                K=K
            )

        else:
            raise ValueError("barrier_mode must be 'hard', 'medium', or 'easy'")

        row['generation'] = int(df_random_original.iloc[gen]['generation'])
        rows.append(row)

        if row['extinction']:
            extinction = True
            break

        prev_x, prev_y = row['final_x'], row['final_y']

    dfBarrier = pd.DataFrame(rows)
    return dfBarrier, dicBarrier, extinction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate shuffle and white-noise CAR control paths from CA reference data."
    )

    parser.add_argument("--reference_root", required=True,
                        help="Root folder containing CA reference paths, e.g. outputs/path_data")
    parser.add_argument("--out_dir", required=True,
                        help="Output folder for generated random-control paths")

    parser.add_argument("--initial_state", default="0001000")
    parser.add_argument("--num_gen", type=int, default=100)

    parser.add_argument("--barrier_gen", type=int, default=49)
    parser.add_argument("--barrier_extend", type=int, default=20)
    parser.add_argument("--barrier_angle", type=int, default=90)
    parser.add_argument("--extra_gen", type=int, default=0)

    parser.add_argument("--rules_start", type=int, default=0)
    parser.add_argument("--rules_end", type=int, default=255)

    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--sources", nargs="+", default=["shuffle", "white_noise"],
                        choices=["shuffle", "white_noise"])
    parser.add_argument("--barrier_modes", nargs="+", default=["easy"],
                        choices=["easy", "medium", "hard"])

    parser.add_argument("--seed", type=int, default=123)

    return parser.parse_args()


def make_output_folders(out_dir, initial_state, num_gen, barrier_gen, barrier_extend, barrier_angle, extra_gen):
    out_root = Path(out_dir) / f"{initial_state}_{num_gen}"
    original_out = out_root / "original"
    barrier_out = out_root / f"gen{barrier_gen}_size{barrier_extend}_angle{barrier_angle}_extra{extra_gen}"

    original_out.mkdir(parents=True, exist_ok=True)
    barrier_out.mkdir(parents=True, exist_ok=True)

    return original_out, barrier_out


def load_reference_original(reference_root, initial_state, num_gen, rule):
    reference_file = (
        Path(reference_root)
        / f"{initial_state}_{num_gen}"
        / "original"
        / f"dfOriginal_r{rule}.csv"
    )

    if not reference_file.exists():
        return None, reference_file

    return pd.read_csv(reference_file), reference_file


def save_barrier_info(dic_barrier, barrier_out, rule, source_type, variant, barrier_mode):
    barrier_info_file = barrier_out / f"dicBarrier_r{rule}_{source_type}_v{variant}_{barrier_mode}.csv"
    pd.DataFrame([dic_barrier]).to_csv(barrier_info_file, index=False)


def main():
    args = parse_args()

    rng = random.Random(args.seed)
    original_out, barrier_out = make_output_folders(
        out_dir=args.out_dir,
        initial_state=args.initial_state,
        num_gen=args.num_gen,
        barrier_gen=args.barrier_gen,
        barrier_extend=args.barrier_extend,
        barrier_angle=args.barrier_angle,
        extra_gen=args.extra_gen,
    )

    expected_original = (
        (args.rules_end - args.rules_start + 1)
        * len(args.sources)
        * args.variants
    )

    expected_barrier = expected_original * len(args.barrier_modes)

    print("\n===== RANDOM CONTROL GENERATION =====")
    print(f"Reference root          : {args.reference_root}")
    print(f"Output root             : {args.out_dir}")
    print(f"Initial state           : {args.initial_state}")
    print(f"Rules                   : {args.rules_start}-{args.rules_end}")
    print(f"Sources                 : {args.sources}")
    print(f"Barrier modes           : {args.barrier_modes}")
    print(f"Variants/rule           : {args.variants}")
    print(f"Random seed             : {args.seed}")
    print(f"Expected original files : {expected_original}")
    print(f"Expected barrier files  : {expected_barrier}")
    print("=====================================\n")

    total_original = 0
    total_barrier = 0
    missing_reference = 0
    errors = 0

    rules = range(args.rules_start, args.rules_end + 1)

    for rule in rules:
        df_reference, reference_file = load_reference_original(
            reference_root=args.reference_root,
            initial_state=args.initial_state,
            num_gen=args.num_gen,
            rule=rule,
        )

        if df_reference is None:
            print(f"[MISSING] reference file not found: {reference_file}")
            missing_reference += 1
            continue

        for source_type in args.sources:
            for variant in range(1, args.variants + 1):
                try:
                    df_random_original = dir_original(
                        df_reference=df_reference,
                        source_type=source_type,
                        variant_num=variant,
                        rng=rng,
                        initial_state=args.initial_state,
                        initial_position=(0, 0),
                    )

                    original_file = original_out / f"dfOriginal_r{rule}_{source_type}_v{variant}.csv"
                    df_random_original.to_csv(original_file, index=False)
                    total_original += 1

                    for barrier_mode in args.barrier_modes:
                        df_barrier, dic_barrier, extinction = dir_barrier(
                            df_reference=df_reference,
                            df_random_original=df_random_original,
                            source_type=source_type,
                            variant_num=variant,
                            barrier_gen=args.barrier_gen,
                            barrier_extend=args.barrier_extend,
                            barrier_angle=args.barrier_angle,
                            barrier_mode=barrier_mode,
                            rng=rng,
                            K=3,
                            initial_state=args.initial_state,
                            initial_position=(0, 0),
                        )

                        barrier_file = barrier_out / f"dfBarrier_r{rule}_{source_type}_v{variant}_{barrier_mode}.csv"
                        df_barrier.to_csv(barrier_file, index=False)

                        save_barrier_info(
                            dic_barrier=dic_barrier,
                            barrier_out=barrier_out,
                            rule=rule,
                            source_type=source_type,
                            variant=variant,
                            barrier_mode=barrier_mode,
                        )

                        total_barrier += 1

                except Exception as e:
                    print(f"[ERROR] rule={rule}, source={source_type}, variant={variant}: {e}")
                    errors += 1

    print("\nDone generating random-control paths.")
    print(f"Original paths generated: {total_original}")
    print(f"Barrier paths generated: {total_barrier}")
    print(f"Missing CA reference files: {missing_reference}")
    print(f"Errors: {errors}")
    print(f"Original output folder: {original_out}")
    print(f"Barrier output folder: {barrier_out}")


if __name__ == "__main__":
    main()