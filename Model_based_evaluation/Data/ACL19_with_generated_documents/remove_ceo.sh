# Navigate through all directories in the current directory
find . -type d -name "CEO" | while read subdir; do
    echo "Removing CEO directory at: $subdir"
    rm -r "$subdir"
    echo "Directory removed."
done
