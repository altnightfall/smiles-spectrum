FROM python:3.11-slim

WORKDIR /app

COPY /frontend /app/.
COPY ./pyproject.toml /app/.
COPY ./.python-version /app/.

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install uv
    
RUN uv sync

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
