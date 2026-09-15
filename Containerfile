FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Pre-fetch the tokenizer at build time so pods start without HF access.
ARG HF_MODEL_ID=LiYuan/amazon-review-sentiment-analysis
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${HF_MODEL_ID}')"

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
