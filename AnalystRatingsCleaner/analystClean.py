import re

# Paths to the raw and cleaned CSV files
input_path = "analyst_ratings_processed.csv"
output_path = "analyst_ratings_fixed.csv"

# Read in all lines from the raw CSV
with open(input_path, 'r', encoding='utf-8') as fin:
    raw_lines = fin.readlines()

fixed_lines = []
for line in raw_lines:
    # Strip only the newline (keep other whitespace)
    stripped = line.rstrip('\n')
    # If the line starts with an integer ID followed by a comma, it's a new record
    if re.match(r'^\d+,', stripped):
        fixed_lines.append(stripped)
    else:
        # Otherwise, it's a continuation of the previous record—append it
        if fixed_lines:
            fixed_lines[-1] += ' ' + stripped
        else:
            fixed_lines.append(stripped)

# Write out the cleaned CSV
with open(output_path, 'w', encoding='utf-8') as fout:
    for l in fixed_lines:
        fout.write(l + '\n')

print(f"Cleaned CSV written to: {output_path}")
