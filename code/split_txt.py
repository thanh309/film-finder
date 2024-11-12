def split_file(input_file, output_without_file_extension, lines_per_file=1000):
    with open(input_file, 'r') as f:
        file_count = 1
        lines = []
        
        for line_num, line in enumerate(f, start=1):
            lines.append(line)
            
            if line_num % lines_per_file == 0:
                output_file = f'{output_without_file_extension}_part{file_count}.txt'
                with open(output_file, 'w') as out_file:
                    out_file.writelines(lines)
                lines = []
                file_count += 1
        
        if lines:
            output_file = f'{output_without_file_extension}_part{file_count}.txt'
            with open(output_file, 'w') as out_file:
                out_file.writelines(lines)


split_file('resources/user_ids.txt', 'resources/ratings/user_ids')