import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from filename import get_replacement_rules

def generate_one_combination(args):
    metal1, metal2, original_lines, base_name, element_line_idx, n_indices, replacement_rules, filename_n_indices, output_dir, rel_path, structure_type = args

    subfolder = os.path.join(output_dir, rel_path, f"{metal1}_{metal2}")
    os.makedirs(subfolder, exist_ok=True)
    local_count = 0

    try:  # error handling for each combination
        # loop through all replacement combinations from 1 to [max_num] N atoms
        for replace_num in range(1, len(n_indices) + 1):
            for combo in itertools.combinations(n_indices, replace_num):
                # map N atom indices to positions in filename to be replaced with 'O'
                filename_replace_positions = []
                for atom_idx in combo:
                    filename_replace_positions.extend(replacement_rules[atom_idx])

                # replace 'N' with 'O' in the filename
                name_chars = list(base_name)
                for pos in set(filename_replace_positions):
                    if pos < len(filename_n_indices):
                        name_chars[filename_n_indices[pos]] = 'O'   # O, S, P, B

                # replace elements in the structure
                # La → metal1, Ce → metal2, N → O (take care with the replacement rules)
                new_elements = [
                    metal1 if e == 'La' else
                    metal2 if e == 'Ce' else
                    'O' if i in combo else e     # O, S, P, B
                    for i, e in enumerate(original_lines[element_line_idx].strip().split())
                ]
                modified_lines = original_lines.copy()
                modified_lines[element_line_idx] = '  '.join(new_elements) + '\n'

                # save output files
                final_name = f"{''.join(name_chars).replace('La', metal1).replace('Ce', metal2)}_{replace_num}O_{metal1}_{metal2}.vasp"
                output_path = os.path.join(subfolder, final_name)
                with open(output_path, 'w') as f:
                    f.writelines(modified_lines)

                local_count += 1
        return (f"{metal1}_{metal2}", local_count, structure_type, None)
    except Exception as e:
        #return (f"{metal1}_{metal2}", local_count, str(e))  
        return (f"{metal1}_{metal2}", local_count, structure_type, str(e))# return the error message if any exception occurs


def generate_structures_parallel(input_dir, output_dir, metals, max_workers, serial, log_file='error_log.txt'):
    all_tasks = []
    structure_counts = {'Co_NNNN_Ce_NNNN_1N': 0, 'Co_NNNN_Ce_NNNN_2N': 0, 'Co_NNNN_Ce_NNNN_di': 0, 
                        'Co_NNNN_N_Ce_NNNN_a': 0, 'Co_NNNN_Ce_NNNN_6': 0, 'Co_NNN_Ce_NNNN_b': 0,
                        'Co_N_Ce_NNNN_c': 0, 'Co_NNNN_Ce_NNNNN_1N1': 0, 'Co_NNNN_Ce_NNNN_2N1':0,
                        'Co_NNNN_N_Ce_NNNN_a1': 0, 'La_NNNNN_Ce_NNNN_d': 0, 'La_NNNN_Ce_NNNN_0':0,
                        'Co_NNN_Ce_NN_e': 0, 'La_NNNN_Ce_NNNN_O6': 0, 'Co_COOO_Ce_CNO_opt': 0,
                        'La_NNNNNNNN_Ce_NNNNNNNN': 0, 'La_NNNNNNNN_Ce_NNNNNNNN_O2': 0, 'La_NNNNNNNN_Ce_NNNNNNNN_leaching': 0}

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith(".vasp"):
                continue
            input_path = os.path.join(root, fname)
            rel_path = os.path.relpath(root, input_dir)
            base_name = fname.replace('.vasp', '')
            full_name = fname.replace('.vasp', '')
            filename_n_indices = [i for i, c in enumerate(base_name) if c == 'N']
            structure_type = base_name

            with open(input_path, 'r') as f:
                original_lines = f.readlines()

            try:
                replacement_rules = get_replacement_rules(base_name)
            except ValueError as e:
                print(f"Skipping {fname}: {e}")
                continue

            element_line_idx = 5
            elements = original_lines[element_line_idx].strip().split()
            n_indices = [i for i, e in enumerate(elements) if e == 'N']

            if len(n_indices) < 1:
                print(f"Skipping {fname}: no N atoms found.")
                continue

            for metal1 in metals:
                for metal2 in metals:
                    if metal1 != metal2:
                        args = (
                            metal1, metal2, original_lines, full_name,
                            element_line_idx, n_indices, replacement_rules,
                            filename_n_indices, output_dir, rel_path, structure_type
                        )
                        all_tasks.append(args)

    total_count = 0  # total model count
    failed = []

    if serial:
        for task in tqdm(all_tasks, desc="Generating structures (serial)..."):
            name, count, structure_type, error = generate_one_combination(task)
            total_count += count
            structure_counts[structure_type] += count
            if error:
                failed.append((name, error))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one_combination, task): task for task in all_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating structures (parallel)..."):
                name, count, structure_type, error = future.result()
                total_count += count
                structure_counts[structure_type] += count
                if error:
                    failed.append((name, error))

    print(f"Total number of structures generated: {total_count}")
    print("\nStructure generation task count by type: ")
    for key, val in structure_counts.items():
        print(f"{key}: {val}")

    if failed:
        print(f"{len(failed)} combinations failed. See {log_file} for details.")
        with open(log_file, 'w') as f:
            for name, err in failed:
                f.write(f"{name} failed: {err}\n")


if __name__ == '__main__':
    generate_structures_parallel(
        input_dir=r"INPUT_PATH",
        output_dir=r"OUTPUT_PATH",
        metals=['Mg', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ce', 'Ni', '  Cu',
                'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'La',
               'Nd', 'Gd', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'],
        max_workers=8,   # adjust based on CPU core count
        serial=False       # set to True for serial execution, False for parallel execution
    )
