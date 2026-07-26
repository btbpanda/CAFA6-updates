project_dir="$(pwd)"

if [[ ! -f "$project_dir/kaggle.json" ]]; then
    echo "Missing Kaggle credentials: $project_dir/kaggle.json" >&2
    exit 1
fi

mkdir -p "$project_dir/data"

docker build -t cafa6-solution .

docker run --gpus all \
    --shm-size=8g \
    --ipc=host \
    -v "$project_dir/kaggle.json:/root/.kaggle/kaggle.json" \
    -v "$project_dir/data:/workspace/data" \
    -it --rm \
    cafa6-solution
