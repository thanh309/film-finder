import os

output_dir = 'resources/data/split_fids'
with open('resources/data/fids_filtered.txt', 'r') as fr:
    total_lines = sum(1 for _ in fr)

os.makedirs(output_dir, exist_ok=True)
lines_per_file = (total_lines + 3) // 4

with open('resources/data/fids_filtered.txt', 'r') as fr:
    for i in range(4):
        output_file = os.path.join(output_dir, f'fids_part_{i + 1}.txt')
        with open(output_file, 'w') as fw:
            for _ in range(lines_per_file):
                line = fr.readline()
                if not line:
                    break
                fw.write(line)