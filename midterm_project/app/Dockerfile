# Use official Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# RUN pip install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ENV PATH="../../venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy

# COPY pyproject.toml uv.lock ./


# RUN uv sync --frozen

COPY predict.py pipeline.py pipeline_v1.bin ./

EXPOSE 9696

# CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]