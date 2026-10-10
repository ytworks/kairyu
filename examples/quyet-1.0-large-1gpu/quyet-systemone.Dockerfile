# syntax=docker/dockerfile:1.7
# This example's System One adapter: the same pinned vLLM image that serves chat,
# plus the quyet package and quyet_systemone.py. The adapter itself runs on the
# CPU; `verify.sh reference` runs the package's own transformers reads from this
# image on the GPU. control.py builds it and attests it by image ID.
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
