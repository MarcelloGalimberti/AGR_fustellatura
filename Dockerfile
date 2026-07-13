FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY OEE_v3.py .
COPY bgt_2026_volumi.xlsx .
COPY Abbinamento_Gruppo.xlsx .
COPY LOGO-Artigrafiche_Italia.png .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "OEE_v3.py", "--server.port=8501", "--server.address=0.0.0.0"]
