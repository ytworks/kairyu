# syntax=docker/dockerfile:1.7
# This example's System One adapter: the same pinned vLLM image that holds the
# model, plus the quyet package and quyet_systemone.py. The adapter itself runs on the
# CPU; `verify.sh reference` runs the package's own transformers reads from this
# image on the GPU. control.py builds it with its source hashes as labels and
# checks those labels, not the image ID (which differs between Docker image stores).
ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}
ARG QUYET_VERSION

COPY quyet-requirements.txt /opt/quyet/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /opt/quyet/requirements.txt \
    && python3 -c "import quyet; assert quyet.__version__ == '${QUYET_VERSION}', quyet.__version__"

COPY quyet_systemone.py /opt/quyet/quyet_systemone.py
RUN python3 -m py_compile /opt/quyet/quyet_systemone.py
WORKDIR /opt/quyet
ENTRYPOINT ["python3", "/opt/quyet/quyet_systemone.py"]
