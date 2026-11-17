FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.12-slim AS build

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=build /app/.venv /app/.venv
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["research-mcp"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
