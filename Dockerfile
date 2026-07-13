FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    albumentations==2.0.8 \
    opencv-python-headless \
    scikit-learn \
    pandas \
    scipy \
    tqdm

WORKDIR /workspace
COPY *.py ./
COPY README.md LICENSE ./

# Data is mounted at runtime:
#   docker run --gpus all -v /path/to/data:/workspace/data freuid \
#     python train.py --model convnext_small --fold all
CMD ["python", "train.py", "--help"]
