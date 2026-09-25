FROM python:3.13-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    sentence-transformers \
    chromadb \
    langgraph \
    fastapi \
    uvicorn \
    pydantic

EXPOSE 7860

ENV MOCK_LLM=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]