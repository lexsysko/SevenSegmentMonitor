ARG PYTHON_VER="3.13"
#ARG PYTHON_VER_UV="-p ${PYTHON_VER}t"
ARG PYTHON_VER_UV=""

######## BUILDER OF PYTHON APP
FROM python:${PYTHON_VER}-slim AS builder

WORKDIR /opt


# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/.venv \
    CC="ccache gcc" \
    CXX="ccache g++" \
    FC="ccache gfortran"

#    UV_PYTHON_DOWNLOADS=never \

COPY pyproject.toml uv.lock ./

ARG PYTHON_VER_UV
ARG VERBOSE

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.cache/ccache \
    uv sync ${PYTHON_VER_UV:-} ${VERBOSE:-} --frozen --no-install-project --no-dev

RUN if [ -n "${PYTHON_VER_UV}" ]; then export PYTHON_GIL=0; fi; ${UV_PROJECT_ENVIRONMENT}/bin/python -c "\
import sys; \
import numpy; \
import cv2; \
print(sys.version); \
print('GIL:', sys._is_gil_enabled()); \
print('numpy:', numpy.__version__); \
print('cv2:', cv2.__version__); \
"

# 1. Base stage used when PYTHON_VER_UV is empty/unset
FROM scratch AS uv_python_source
WORKDIR /uv-python

# 2. Stage used ONLY when PYTHON_VER_UV is set
FROM builder AS uv_python_source_set
RUN mkdir -p /uv-python \
 && cp -a /root/.local/share/uv/python/. /uv-python/ 2>/dev/null || true

# 3. Dynamic target selection (SINGLE LINE)
# Unset/Empty -> targets "uv_python_source"
# Set          -> targets "uv_python_source_set"
FROM uv_python_source${PYTHON_VER_UV:+_set} AS uv_python_final

FROM python:${PYTHON_VER}-slim AS runner

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

#ARG _USER=appuser
#ARG _GROUP=appgroup
#RUN groupadd ${_GROUP} && useradd --no-log-init -r --no-create-home -g ${_GROUP} ${_USER} && \
#    mkdir ./data && \
#    chown -R  ${_USER}:${_GROUP} ./data


# Copy venv from previous stage "builder"
COPY --from=uv_python_final /uv-python /root/.local/share/uv/python
COPY --from=builder /opt/.venv /opt/.venv
COPY pyproject.toml .
COPY ./src ./src/
COPY --chmod=+x ./entrypoint.sh .

ENV PATH="/opt/.venv/bin:$PATH" JACK_NO_START_SERVER=1 LIBJACK_NO_AUDIO_CRITICAL=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

# RUN printf 'pcm.!default {\n  type hw\n  card 1\n  device 0\n}\nctl.!default {\n  type hw\n  card 1\n}\n' > /etc/asound.conf

#USER ${_USER}

CMD ["/app/entrypoint.sh"]

