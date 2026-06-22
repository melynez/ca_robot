"""
generate_car_path.py

Generates:

1. Original CA trajectories
   dfOriginal_r{rule}.csv

2. Barrier CA trajectories
   dfBarrier_r{rule}.csv

3. Barrier metadata
   dicBarrier_r{rule}.csv

for elementary cellular automata rules 0-255.

Outputs are organized by:

<initial_state>_<num_gen>/
    original/
    gen{barrier_gen}_size{barrier_extend}_angle{barrier_angle}_extra{extra_gen}/
"""

import argparse
import math
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
def str_to_dir(rule_num, initial_state, num_gen, initial_position=(0, 0)):
    # Define direction mappings
    dicDirection = {'00': (0, 1), '01': (1, 0), '10': (0, -1), '11': (-1, 0)}
    dfOriginal = pd.DataFrame(columns=['generation', 'bin_string', 'final_x', 'final_y'])
    
    dicRule = rule_to_string(rule_num)
    x_init, y_init = initial_position
    
    current_state = initial_state
    bg_type = '0'  # Initial background type
    
    dtype = [
        ('generation', 'i4'),       # Integer for generation
        ('bin_string', 'U256'),  # Unicode string
        ('final_x', 'f4'),         # Float for x-coordinate
        ('final_y', 'f4')          # Float for y-coordinate
    ]
    og_array = np.zeros(num_gen, dtype=dtype)
    og_array[0] = (0, current_state, x_init, y_init)

    # dfOriginal.loc[0] = [0, current_state, x_init, y_init]

    # Iterate through each generation
    for gen in range(1, num_gen):
        expanded_state = bg_type * 2 + current_state + bg_type * 2
        new_state = ''.join(str(dicRule[expanded_state[i:i+3]]) for i in range(len(expanded_state)-2))
        # print(f'expanded prev: {expanded_state}')
        # print(f'new_state: {new_state}')

        # Determine new background type (integrated here)
        bg_type = str(dicRule[bg_type * 3])

        # Initialize the current DataFrame for tracking positions and movements

        # Start from the previous generation's final position
        temp_x, temp_y = og_array['final_x'][gen-1], og_array['final_y'][gen-1]
        
        # Iterate through the binary string in 2-bit chunks and apply movements
        # for i in range(0, len(new_state)-1):
        #     two_bit = new_state[i:i + 2]
        #     delta_x, delta_y = dicDirection.get(two_bit, (0, 0))
        #     temp_x += delta_x
        #     temp_y += delta_y
        
        for i in range(len(new_state)):
            # If it's the last bit, pair it with the first bit
            if i == len(new_state) - 1:
                two_bit = new_state[i] + new_state[0]
            else:
                two_bit = new_state[i:i + 2]
            
            delta_x, delta_y = dicDirection.get(two_bit, (0, 0))
            temp_x += delta_x
            temp_y += delta_y

            # print(f'\ndelta: {delta_x,delta_y}')
            # print(f'coord: {temp_x,temp_y}')


        # Update the final position for the current generation
        x, y = temp_x, temp_y

        og_array[gen] = (gen, new_state, temp_x, temp_y)

        current_state = new_state
        
    dfOriginal = pd.DataFrame(og_array)
    return dfOriginal



def dir_barrier(rule_num, initial_state, num_gen, dicBarrier, dfOriginal, barrier_gen, initial_position=(0, 0)):

    barrier_set = False
    cross_gen = None
    extra_point = num_gen + 1
    num_gen = num_gen + extra_gen
    extra = None # initialize this for now
    extinction = None

    def cross_test(p1, p2, p3, p4):
        epsilon=0.009

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate the denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # If den is close to zero, consider lines as parallel
        if abs(den) < epsilon:
            return False
        
        # Calculate the intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        # Check if intersection point is on or very close to both line segments
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


    dicDirection = {'00': (0, 1), '01': (1, 0), '10': (0, -1), '11': (-1, 0)}
    dfBarrier = pd.DataFrame(columns=['generation', 'bin_string', 'final_x', 'final_y', 'extra_gen', 'after_barrier'])
    dicRule = rule_to_string(rule_num)
    x_init, y_init = initial_position
   
    barrier_point = dfOriginal.iloc[barrier_gen]['final_x'], dfOriginal.iloc[barrier_gen]['final_y']
    print(f"THE BARRIER POINT IS: {barrier_point}")

    dtype = [
        ('gen', 'i4'),
        ('bin_string', 'O'),
        ('final_x', 'f4'),
        ('final_y', 'f4'),
        ('extra_gen', '?'),
        ('after_barrier', '?'),
        ('attempts_used', 'i4'),
        ('bits_removed', 'i4')
    ]

    current_state = initial_state 
    barrier_array = np.zeros(num_gen, dtype=dtype)
    barrier_array[0] = (0, current_state, x_init, y_init, False, False, 0, 0)
    
    # initialize final_x & final_y (1st generation's previous)
    final_x, final_y = x_init, y_init
    bg_type = '0' 

    for gen in range(1, num_gen):

        if gen < extra_point:
            extra = False 
        else:
            extra = True

        attempts_used = 0
        bits_removed = 0

        if gen >= barrier_gen - 1:
            attempts_used = 1

        if gen >= barrier_gen:  # If we haven't detected a barrier crossing yet
            after_barrier = True
        else:
            after_barrier = False  # Every gen after the first crossing


        # find prev_state w/ bin_string (index 1 in numpy array)
        prev_state = barrier_array['bin_string'][gen-1]
        # prev_state = barrier_array[gen-1, 1]
        expanded_state = bg_type * 2 + prev_state + bg_type * 2
        
        new_state = ''.join(str(dicRule[expanded_state[i:i+3]]) for i in range(len(expanded_state)-2))
        bg_type = str(dicRule[bg_type * 3])

        prev_x, prev_y = barrier_array['final_x'][gen-1], barrier_array['final_y'][gen-1]
        temp_x, temp_y = prev_x, prev_y

        # iterate to find final_x & final_y for this generation (2-bits by 2-bits)
        # use this -> is there POI w/ barrier here?
        for m in range(len(new_state)):
            # If it's the last bit, pair it with the first bit
            if m == len(new_state) - 1:
                two_bit = new_state[m] + new_state[0]
            else:
                two_bit = new_state[m:m + 2]
            
            delta_x, delta_y = dicDirection.get(two_bit, (0, 0))
            temp_x += delta_x
            temp_y += delta_y

        # for m in range(0, len(new_state) - 1):
        #     two_bit = new_state[m:m + 2]
        #     delta_x, delta_y = dicDirection.get(two_bit, (0, 0))
        #     temp_x += delta_x
        #     temp_y += delta_y
        
        final_x, final_y = temp_x, temp_y 
        final_state = new_state
        barrier_array[gen] = (gen, final_state, final_x, final_y, extra, after_barrier, attempts_used, bits_removed)
        
        # variables used ONLY for barrier calculation
        temp_fullx = final_x
        temp_fully = final_y
        prev_fullx = barrier_array['final_x'][gen-1]
        prev_fully = barrier_array['final_y'][gen-1]

        # user input barrier info as variables -- use helper functions easily 
        barrier_extend = dicBarrier['barrier_extend']
        angle_degrees = dicBarrier['barrier_angle']

        if temp_fullx is not None and temp_fully is not None:
            if prev_fullx != temp_fullx:
                ca_slope = (prev_fully - temp_fully) / (prev_fullx - temp_fullx)
            else:
                ca_slope = 0  # Treat as a horizontal path when x-values are the same

        if not barrier_set:
            barrier_start, barrier_end = calculate_barrier(barrier_point, ca_slope, barrier_extend, angle_degrees)
            print(f"generation: {gen}. points: {barrier_start, barrier_end}.")
        
        print("Here is: (temp_fullx, temp_fully),  (prev_fullx, prev_fully),  barrier_start,   barrier_end)")
        print(f"{(temp_fullx, temp_fully)}, {(prev_fullx, prev_fully)}, {barrier_start}, {barrier_end}")
        # if there is POI w/ current barrier
        if gen >= barrier_gen - 1 and cross_test((temp_fullx, temp_fully), 
                      (prev_fullx, prev_fully),
                      barrier_start, barrier_end):
            
            if cross_gen is None:  # First crossing detected
                cross_gen = gen  # Store first crossing generation
                dicBarrier['barrier_start'] = barrier_start
                dicBarrier['barrier_end'] = barrier_end

                barrier_set = True
                print(f"JOB SUCCESS {barrier_start, barrier_end}")


            after_barrier = True  # Set flag for all future generations
            barrier_array['after_barrier'][gen] = after_barrier


            print(f"Cross detected in generation {gen}.")
            
            # init variables -> electric shock 
            dicDirection = {'00': (0, 1), '01': (1, 0), '10': (0, -1), '11': (-1, 0)}
            flagged_bits = [] # 2nd bit of 2-bits that cause POI
            i1 = 0 # index of 1st bit of 2-bit (i determines the rows, each 2-bit ID by its i)
            num_cross = 0 # for each 2-bit round, # of times cross for each i 

            current_array = np.zeros((len(new_state), 5), dtype=object)
            current_index = 0
            n = len(new_state)




            # Step 1: Flag bits
            while i1 <= n - 1:
                i2 = i1 + 1 + num_cross  # Compute index of the second bit dynamically

                if i2 == len(new_state): # wrap around at final bit (+1, -1)
                    i2 = 0 # i2 = 1st bit, i1 = "nth" edge bit
                elif i2 > len(new_state):
                    break


                bit1 = new_state[i1]
                bit2 = new_state[i2]
                two_bit = bit1 + bit2
                delta_x, delta_y = dicDirection.get(two_bit, (0, 0))

                # Calculate current xy coordinates as temp_2bx & temp_2by using previous gen
                if i1 == 0:
                    # For the very first 2 bits, use the previous generation's final coordinates
                    temp_2bx = prev_fullx + delta_x
                    temp_2by = prev_fully + delta_y
                else:
                    # For subsequent 2 bits, use the most recent temp coordinates 
                    if current_index > 0:
                        temp_2bx = current_array[current_index-1, 3] + delta_x
                        temp_2by = current_array[current_index-1, 4] + delta_y
                    else:
                        temp_2bx = prev_fullx + delta_x
                        temp_2by = prev_fully + delta_y



                if cross_test((prev_x, prev_y), 
                              (temp_2bx, temp_2by),
                              barrier_start, 
                              barrier_end):

                    num_cross += 1 
                    # print(f"processing index #{i} as i1")

                    print(f"prev_xy: {(prev_x, prev_y)}")
                    # print(f"i1 (1st index): {i}")
                    # print(f"i2 (2nd index): {n}")
                    print(f"num_cross for {two_bit} in {gen} is {num_cross}")
                    print(f"p1-p4: {(prev_x, prev_y)}, {(temp_2bx, temp_2by)}, {barrier_start}, {barrier_end}")

                    flagged_bits.append(i2) 
                else:
                    # print(f"processing index #{i} as i1")
                   # add calculated temp_2bx/y (the x,y after applying ONLY 2-bits) to current_array
                    current_array[current_index] = [two_bit, delta_x, delta_y, temp_2bx, temp_2by]
                    current_index += 1

                    # Move to the next starting bit
                    i1 += 1 + num_cross
                    num_cross = 0  # Reset crossing counter for the next pair
                        





            print(flagged_bits) # ignore: not used in code.

            # Step 2: Create the final string by removing all flagged bits
            final_state = ''.join(bit for k, bit in enumerate(new_state) if k not in flagged_bits)
            
            print(f"final_state: {final_state}.")           
            print(f"new_state: {new_state}.")           


            if final_state == '':
                extinction = True
                barrier_array[gen] = (gen, '', final_x, final_y, extra, after_barrier, attempts_used, bits_removed)
                extinction_gen = gen
                print(f"Extinction occurred at generation {extinction_gen}")
                break  # Stop evolving, but keep the data up to this point

            if final_state == '0':
                barrier_array[gen] = (gen, '', final_x, final_y, extra, after_barrier, attempts_used, bits_removed)
                extinction_gen = gen

                print(f"Extinction occurred at generation {extinction_gen}")


            if current_index == 0:
                print("Warning: current_array is empty")
                break

            # Step 3: recalculate (x,y) from final_state -- add to dfBarrier
            
            # Initialize starting position
            final_temp_x, final_temp_y = prev_x, prev_y
            

            final_x = current_array[current_index-1, 3]
            final_y = current_array[current_index-1, 4]

            barrier_array['bin_string'][gen] = final_state
            barrier_array['final_x'][gen] = final_x
            barrier_array['final_y'][gen] = final_y

            # banana; bits removed.
            bits_removed = len(new_state) - len(final_state)
            barrier_array['bits_removed'][gen] = bits_removed
            
            print(f"this gen x: {barrier_array['final_x'][gen]}")
            print(f"this gen y: {barrier_array['final_y'][gen]}")
            print(f"prev gen x: {barrier_array['final_x'][gen-1]}")
            print(f"prev gen y: {barrier_array['final_y'][gen-1]}")

            print(f'ice cream {final_state}')
            print(f'current_array: {current_array[current_index-1, 3], current_array[current_index-1, 4]}')
            print(f'final_temp_xy (recalc): {final_temp_x, final_temp_y}')

        else:
            barrier_array['bin_string'][gen] = final_state
            barrier_array['final_x'][gen] = final_x
            barrier_array['final_y'][gen] = final_y
            barrier_array['after_barrier'][gen] = after_barrier
            barrier_array['attempts_used'][gen] = attempts_used
            barrier_array['bits_removed'][gen] = bits_removed
            
            print("No crossing was detected.")
            # barrier_start, barrier_end = (None, None)



    dfBarrier = pd.DataFrame(barrier_array)
    
    # Remove rows where gen, bin_string, final_x, and final_y are all zero, except for the 0th row
    mask = (dfBarrier['gen'] == 0) & (dfBarrier['bin_string'] == 0) & (dfBarrier['final_x'] == 0) & (dfBarrier['final_y'] == 0)
    dfBarrier = dfBarrier.loc[~mask | (dfBarrier.index == 0)].reset_index(drop=True)

    # Reset the index
    dfBarrier = dfBarrier.reset_index(drop=True)


    print(f"barrier_start & end is {barrier_start, barrier_end}")
    print(f"nomnom{dicBarrier}")
    return dfBarrier, dicBarrier, extinction










def generate_all_7bit_states():
    return [format(i, '07b') for i in range(128)]




def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--initial_state", default="0001000")
    ap.add_argument("--num_gen", type=int, default=100)
    ap.add_argument("--barrier_gen", type=int, default=49)
    ap.add_argument("--barrier_extend", type=int, default=20)
    ap.add_argument("--barrier_angle", type=int, default=90)
    ap.add_argument("--extra_gen", type=int, default=0)
    ap.add_argument("--rules_start", type=int, default=0)
    ap.add_argument("--rules_end", type=int, default=255)
    ap.add_argument("--workers", type=int, default=1)
    return ap.parse_args()



def create_folder_structure(base_path, initial_state, num_gen, barrier_gen, barrier_extend, barrier_angle, extra_gen):
    main_folder = base_path / f"{initial_state}_{num_gen}"
    original_folder = main_folder / "original"
    barrier_folder = main_folder / f"gen{barrier_gen}_size{barrier_extend}_angle{barrier_angle}_extra{extra_gen}"

    main_folder.mkdir(parents=True, exist_ok=True)
    original_folder.mkdir(exist_ok=True)
    barrier_folder.mkdir(exist_ok=True)

    return main_folder, original_folder, barrier_folder

# Function to process a single CA rule
def process_ca_rule(rule_num, barrier_gen, barrier_extend, barrier_angle, initial_state, num_gen, extra_gen, initial_position, base_dir):
    # Create a directory for this rule
    main_folder, original_folder, barrier_folder = create_folder_structure(
        base_dir, initial_state, num_gen, barrier_gen, barrier_extend, barrier_angle, extra_gen
    )

    # Generate original CA path
    dfOriginal = str_to_dir(rule_num, initial_state, num_gen, initial_position)
    dfOriginal.to_csv(original_folder / f'dfOriginal_r{rule_num}.csv', index=False)

    # Extract barrier point from generation barrier_gen
    barrier_point = (dfOriginal.loc[barrier_gen, 'final_x'], dfOriginal.loc[barrier_gen, 'final_y'])


    # Create barrier dictionary
    dicBarrier = {
        'barrier_extend': barrier_extend,
        'barrier_angle': barrier_angle, 
    }

    # Generate CA path with barrier
    dfBarrier, dicBarrierEnds, extinction = dir_barrier(
        rule_num, initial_state, num_gen, dicBarrier, dfOriginal, barrier_gen, initial_position
    )



    # # Generate original plots
    # plot_files_original = plot_pathways(
    #     dfOriginal, pd.DataFrame(), dicBarrier, rule_num, initial_state, num_gen,
    #     None, None, extinction=False, extra_gen=extra_gen, barrier_folder=barrier_folder, original_folder=original_folder
    # )

    # plot_files = plot_pathways(
    #     dfOriginal, dfBarrier, dicBarrierEnds, rule_num, initial_state, num_gen,
    #     barrier_point, barrier_extend, extinction, extra_gen, barrier_folder, original_folder
    # )


    # Save barrier data
    dfBarrier.to_csv(barrier_folder / f'dfBarrier_r{rule_num}.csv', index=False)
    
    # Merge dicBarrier & start/end points
    dicBarrier = dicBarrier | dicBarrierEnds

    # Save dicBarrier as a CSV file
    dicBarrier_df = pd.DataFrame([dicBarrier])
    dicBarrier_df.to_csv(barrier_folder / f'dicBarrier_r{rule_num}.csv', index=False)


    return {
        'dfOriginal': dfOriginal,
        'dfBarrier': dfBarrier,
        'plots': None, # plot_files
        'dicBarrier': dicBarrier
    }


import multiprocessing as mp


def process_rule_wrapper(args):
    return process_ca_rule(*args)


    
if __name__ == '__main__':

    args = parse_args()

    initial_states = [args.initial_state]

    num_gen = args.num_gen
    extra_gen = args.extra_gen

    initial_position = (0, 0)

    barrier_angle_values = [args.barrier_angle]
    barrier_gen_values = [args.barrier_gen]
    barrier_extend_values = [args.barrier_extend]

    selected_rules = list(
        range(args.rules_start,
              args.rules_end + 1)
    )

    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== CA PATH GENERATION =====")
    print(f"Output root     : {base_dir}")
    print(f"Initial state   : {args.initial_state}")
    print(f"Rules           : {args.rules_start}-{args.rules_end}")
    print(f"Barrier gen     : {args.barrier_gen}")
    print(f"Barrier extend  : {args.barrier_extend}")
    print(f"Barrier angle   : {args.barrier_angle}")
    print(f"Workers         : {args.workers}")
    print("==============================\n")

    tasks = []

    for initial_state in initial_states:
        for barrier_gen in barrier_gen_values:
            for barrier_extend in barrier_extend_values:
                for barrier_angle in barrier_angle_values:
                    for rule in selected_rules:

                        tasks.append(
                            (
                                rule,
                                barrier_gen,
                                barrier_extend,
                                barrier_angle,
                                initial_state,
                                num_gen,
                                extra_gen,
                                initial_position,
                                base_dir
                            )
                        )

    if args.workers == 1:

        results = [
            process_rule_wrapper(t)
            for t in tasks
        ]

    else:

        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(
                process_rule_wrapper,
                tasks
            )

    print(f"Finished {len(results)} rule runs.")

