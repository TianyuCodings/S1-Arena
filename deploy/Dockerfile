# Two images for sweeps on a cluster or in the cloud (see docs/deploy.md). Build from the repository root:
#   docker build -f deploy/Dockerfile --target core   -t s1arena:core .
#   docker build -f deploy/Dockerfile --target balrog -t s1arena:balrog .
# core:   static tasks and all environments except the BALROG games, WebShop (a separate server) and the
#         Docker-per-seed coding controller (`s1arena sweep-docker`, run on a host with Docker).
# balrog: game/open_world, game/rule_puzzles, game/roguelike (Crafter, Baba Is AI, MiniHack, NetHack), whose pins
#         (numpy 1.26, gym 0.23, NLE built from source) do not fit in the core image. Mirrors
#         scripts/setup/balrog/setup_balrog_env.sh, with the clock-pinning NLE patch applied.
# No secrets are baked in: pass AI_GATEWAY_API_KEY (and HF_TOKEN) at run time.

FROM python:3.12-slim-bookworm AS base
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY s1arena ./s1arena

FROM base AS core
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
# pip ignores [tool.uv.sources]: pin tau2 to the git commit, or PyPI's unrelated `tau2` gets installed
RUN pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@b7ea9074c1cba482b30687fecdb5c8425fd6f619" \
    && pip install -e '.[envs,alfworld]' jericho==3.3.1 swebench==5.0.2 loguru \
    && python -m spacy download en_core_web_sm
# ScienceWorld at commit e8216d6 (PyPI 1.2.3 is not deterministic across JVM launches)
RUN git clone https://github.com/allenai/ScienceWorld /opt/ScienceWorld \
    && git -C /opt/ScienceWorld checkout e8216d6 && pip install /opt/ScienceWorld
# the Docker CLI only, for `s1arena sweep-docker` with the host's socket mounted (-v /var/run/docker.sock:...)
RUN curl -sL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
    | tar xz -C /usr/local/bin --strip-components=1 docker/docker
COPY data ./data
COPY sweeps ./sweeps
COPY scripts ./scripts
ENTRYPOINT ["s1arena"]

FROM base AS balrog
RUN apt-get update && apt-get install -y --no-install-recommends cmake flex bison && rm -rf /var/lib/apt/lists/*
COPY scripts/setup/balrog/nle_clockpin.patch.py /tmp/nle_clockpin.patch.py
# balrog-nle has wheels for CPython <= 3.11 only; build it from source (its setup.py asks git for a commit: shim it)
RUN mkdir -p /tmp/shims && printf '#!/bin/sh\nexit 128\n' > /tmp/shims/git && chmod +x /tmp/shims/git \
    && cd /tmp && curl -sL https://files.pythonhosted.org/packages/source/b/balrog-nle/balrog_nle-0.9.0.tar.gz | tar xz \
    && python /tmp/nle_clockpin.patch.py /tmp/balrog_nle-0.9.0 /tmp/balrog_nle-clockpin \
    && cd /tmp/balrog_nle-clockpin && PATH=/tmp/shims:$PATH CMAKE_POLICY_VERSION_MINIMUM=3.5 MAKEFLAGS=-j8 \
       pip wheel --no-deps -w /tmp/wheels . \
    && rm -rf /tmp/balrog_nle-0.9.0 /tmp/balrog_nle-clockpin
RUN PATH=/tmp/shims:$PATH pip install --find-links /tmp/wheels -e . \
        "numpy==1.26.4" "gym==0.23.0" "setuptools<81" "crafter==1.8.3" \
        "baba @ https://github.com/nacloos/baba-is-ai/archive/40ae1b29dbc67f2ecea06cc161ed5ca51719e5e3.tar.gz" \
        "balrog-nle==0.9.0" \
        "minihack @ https://github.com/balrog-ai/minihack/archive/3ecb6da4eadcac7a4dbc5f8e801ccf636e34d7ec.tar.gz" \
        "minigrid>=3.0" "gymnasium>=1.1" \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless "numpy==1.26.4" \
    && python -c "import crafter, baba, nle, minihack, gym; from nle import nle_language_obsv"
COPY data ./data
COPY sweeps ./sweeps
COPY scripts ./scripts
ENTRYPOINT ["s1arena"]
