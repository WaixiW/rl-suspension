ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system suspension && useradd --system --gid suspension suspension
WORKDIR /opt/rl-suspension

COPY pyproject.toml README.md ./
COPY deploy/production/constraints.lock ./deploy/production/constraints.lock
COPY rl_suspension ./rl_suspension
RUN python -m pip install --no-cache-dir \
    --constraint deploy/production/constraints.lock ".[export]"

RUN mkdir -p /opt/rl-suspension/runs /opt/rl-suspension/config \
    && chown -R suspension:suspension /opt/rl-suspension
USER suspension

ENTRYPOINT ["python", "-m", "rl_suspension.production.cli"]
CMD ["--help"]
