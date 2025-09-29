#!/bin/bash
# Combines all Python files in the current directory into combined.py

output="combined.py"

for file in *.py; do
    echo "### File: $file" >> "$output"
    cat "$file" >> "$output"
    echo -e "\n" >> "$output"
done

echo "All Python files combined into $output"
