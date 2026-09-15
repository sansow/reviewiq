"""KServe v2 REST client for the ReviewIQ sentiment model served by OVMS on RHOAI.

OVMS serves the raw ONNX graph, so tokenization and softmax live here.
Model: LiYuan/amazon-review-sentiment-analysis (BERT, 5 classes = 1-5 stars).
"""

import os
from functools import lru_cache

import httpx
import numpy as np
from transformers import AutoTokenizer

MODEL_ID = os.getenv("HF_MODEL_ID", "LiYuan/amazon-review-sentiment-analysis")
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "reviewiq-sentiment")
VERIFY_TLS = os.getenv("VERIFY_TLS", "true").lower() != "false"
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "256"))
BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "16"))


class InferenceNotConfigured(Exception):
    pass


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _v2_tensor(name: str, arr: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(arr.shape),
        "datatype": "INT64",
        "data": arr.flatten().tolist(),
    }


def _infer_batch(texts: list[str], client: httpx.Client) -> np.ndarray:
    tok = get_tokenizer()
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="np",
    )
    inputs = [
        _v2_tensor("input_ids", enc["input_ids"].astype(np.int64)),
        _v2_tensor("attention_mask", enc["attention_mask"].astype(np.int64)),
    ]
    if "token_type_ids" in enc:
        inputs.append(_v2_tensor("token_type_ids", enc["token_type_ids"].astype(np.int64)))

    url = f"{INFERENCE_ENDPOINT}/v2/models/{MODEL_NAME}/infer"
    resp = client.post(url, json={"inputs": inputs})
    resp.raise_for_status()
    body = resp.json()

    out = next((o for o in body["outputs"] if o["name"] == "logits"), body["outputs"][0])
    logits = np.array(out["data"], dtype=np.float32).reshape(out["shape"])
    return _softmax(logits)


def predict(texts: list[str]) -> list[dict]:
    """Return [{stars, confidence, distribution: {1..5: p}}] per input text."""
    if not INFERENCE_ENDPOINT:
        raise InferenceNotConfigured(
            "INFERENCE_ENDPOINT is not set. Point it at your RHOAI model route, "
            "e.g. https://reviewiq-sentiment-predictor-reviewiq.apps.<cluster>."
        )
    results: list[dict] = []
    with httpx.Client(verify=VERIFY_TLS, timeout=60.0) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            probs = _infer_batch(texts[i : i + BATCH_SIZE], client)
            for row in probs:
                stars = int(np.argmax(row)) + 1
                results.append(
                    {
                        "stars": stars,
                        "confidence": round(float(row.max()), 4),
                        "distribution": {str(s + 1): round(float(p), 4) for s, p in enumerate(row)},
                    }
                )
    return results


def ready() -> dict:
    """Check the model server without running inference."""
    if not INFERENCE_ENDPOINT:
        return {"configured": False, "ready": False}
    try:
        with httpx.Client(verify=VERIFY_TLS, timeout=5.0) as client:
            r = client.get(f"{INFERENCE_ENDPOINT}/v2/models/{MODEL_NAME}/ready")
            return {"configured": True, "ready": r.status_code == 200}
    except httpx.HTTPError:
        return {"configured": True, "ready": False}
