FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    timm==1.0.27 \
    albumentations==2.0.8 \
    opencv-python-headless \
    scikit-learn \
    pandas \
    scipy \
    tqdm

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /workspace
COPY *.py ./
COPY README.md LICENSE ./

# Sandbox contract (see README):
#   docker run --network none --gpus all \
#     -v /path/to/images:/data:ro \
#     -v /path/to/weights:/weights:ro \
#     -v /path/to/out:/submissions \
#     freuid
# Weights: https://www.kaggle.com/datasets/ypsrathore/freuid-2026-pratap424-weights
# (files: convnext_small_f0..f4.pth, eva02_large_f0.pth)
# Output: /submissions/submission.csv  (id,label)
CMD ["python", "docker_infer.py"]
