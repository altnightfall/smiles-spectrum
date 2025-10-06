FROM python:3.11-slim

WORKDIR /app

COPY /backend /app/.
COPY ./pyproject.toml /app/.

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install uv
    
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
