FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CAMELTOOLS_DATA=/opt/camel_tools

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cargo \
        cmake \
        curl \
        espeak-ng \
        git \
        libboost-all-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY nabra/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p "$CAMELTOOLS_DATA" \
    && camel_data -i disambig-mle-calima-msa-r13

COPY nabra/ .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
