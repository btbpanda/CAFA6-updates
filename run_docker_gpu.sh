docker build -t cafa6-solution .

docker run --gpus all \
    --shm-size=8g \
    --ipc=host \
    -v $(pwd)/kaggle.json:/root/.kaggle/kaggle.json \
    -v $(pwd)/data:/workspace/data \
    -it --rm \
    cafa6-solution
