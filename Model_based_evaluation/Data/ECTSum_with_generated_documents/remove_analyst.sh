#!/bin/bash

# Navigate through all directories in the current directory
find . -type d | while read dir; do
    # Find all text files containing "analyst" in their names within each directory
    find "$dir" -type f -name '*analyst*.txt' -exec rm {} \;
done

echo "All specified files have been removed."

