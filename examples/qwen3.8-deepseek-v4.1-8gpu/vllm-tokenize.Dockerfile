# The wrapper verifies this local parent's immutable image ID before building.
ARG VLLM_BASE_IMAGE=local/vllm-openai:deepseek-v41-sm120
FROM ${VLLM_BASE_IMAGE}
COPY tokenize-native-chat.patch /opt/kairyu/tokenize-native-chat.patch
RUN echo "0bb4ddf2c0b1412e88c9d853e2ef7163f62665df0b8bbef86df5bdbdd9a76e44  /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/tokenize/protocol.py" | sha256sum -c - \
 && patch --directory=/usr/local/lib/python3.12/dist-packages -p1 --batch --fuzz=0 < /opt/kairyu/tokenize-native-chat.patch \
 && echo "77672d5daeafbc1039f5989e2168e6cda88bc6df1ea3fe6338a44eb15fcc31ee  /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/tokenize/protocol.py" | sha256sum -c -
