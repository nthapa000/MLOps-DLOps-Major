FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers==4.40.2 \
    torch==2.2.2 \
    sentencepiece==0.2.0 \
    protobuf==4.25.3 \
    nltk==3.8.1 \
    numpy==1.26.4 \
    sacrebleu==2.4.2

# Download NLTK tokenizer data (punkt_tab required by nltk >= 3.8.1)
RUN python -m nltk.downloader punkt punkt_tab

RUN mkdir -p /app/dataset

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "translate.py"]
