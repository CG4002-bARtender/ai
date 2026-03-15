FROM xilinx/vitis-ai-pytorch-cpu:latest

RUN /opt/vitis_ai/conda/bin/conda run -n vitis-ai-pytorch \
    pip install librosa scipy scikit-learn matplotlib

COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY arch/ /opt/vitis_ai/compiler/arch/

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src \
    CONDA_DEFAULT_ENV=vitis-ai-pytorch \
    PATH=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib
ENV LD_LIBRARY_PATH=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib:$LD_LIBRARY_PATH

CMD ["bash", "-c", "python scripts/quantize.py && python scripts/compile.py"]
