import os
from collections import defaultdict

def process_poscar(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))

    coord_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(("direct", "cartesian")):
            coord_start = i
            coord_type = line.strip()
            break

    if coord_start is None:
        print(f"no Cartesian:{file_path}")
        return

    coords = lines[coord_start+1:]

    # sum atom num.
    atom_dict = defaultdict(list)
    idx = 0
    for atom, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atom_dict[atom].append(coords[idx])
            idx += 1

    # combine the atom
    new_atom_types = []
    new_atom_counts = []
    new_coords = []
    for atom in sorted(atom_dict.keys()):  # orden
        new_atom_types.append(atom)
        new_atom_counts.append(len(atom_dict[atom]))
        new_coords.extend(atom_dict[atom])

    # replace
    lines[5] = "  ".join(new_atom_types) + "\n"
    lines[6] = "  ".join(map(str, new_atom_counts)) + "\n"
    lines = lines[:coord_start+1] + new_coords
    with open(file_path, "w") as f:
        f.writelines(lines)

    print(f"done: {file_path}")


def process_folder(folder_path):
    valid_names = ("poscar", "contcar") 
    valid_exts = (".vasp")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(name in file.lower() for name in valid_names) or file.lower().endswith(valid_exts):  
                file_path = os.path.join(root, file)
                process_poscar(file_path)


if __name__ == "__main__":
    folder = r"path"  # folder path, no file name
    process_folder(folder)
