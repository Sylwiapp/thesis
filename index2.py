import re

def extract_first_segment_vmrk(vmrk_file, output_file):
    with open(vmrk_file, 'r') as file:
        lines = file.readlines()

    marker_section = False
    segment_start = None
    segment_end = None
    output_lines = []

    for line in lines:
        if "[Marker Infos]" in line:
            marker_section = True
            output_lines.append(line)
            continue

        if not marker_section:
            output_lines.append(line)
        else:
            match = re.match(r'Mk\d+=Response,M \d+,(?P<position>\d+),', line)
            if match:
                if segment_start is None:
                    segment_start = int(match.group("position"))
                elif segment_end is None:
                    segment_end = int(match.group("position"))
                    break

    if segment_start is None or segment_end is None:
        raise ValueError("Nie znaleziono wystarczającej liczby znaczników M w pliku.")

    for line in lines:
        match = re.match(r'Mk\d+=\w+,,(?P<position>\d+),', line)
        if match:
            position = int(match.group("position"))
            if segment_start <= position <= segment_end:
                output_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(output_lines)

# Przykład użycia
vmrk_file = '/home/syl/repo/master/IDTxl/sub-ARZ000_task_art_watch2_run-01.vmrk'
output_file = '/home/syl/repo/master/IDTxl/sub-ARZ000_task_art_watch2_run-01-01.vmrk'
extract_first_segment_vmrk(vmrk_file, output_file)

print(f"Nowy plik .vmrk został zapisany jako {output_file}")
