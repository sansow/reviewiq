# ReviewIQ

AI-powered Amazon review intelligence on OpenShift + OpenShift AI.

Reviews go in, a BERT model (`LiYuan/amazon-review-sentiment-analysis`) served
by OVMS on RHOAI scores them 1–5 stars, and the dashboard shows the sentiment
shape: distribution, negative share, and what positive vs negative reviews
keep mentioning.

## Architecture

```
Browser ── Route ── FastAPI (app/)  ── KServe v2 REST ── OVMS InferenceService (RHOAI)
                    tokenize + softmax                    ONNX model from S3/MinIO
                    aggregate + themes
```

The app container holds the tokenizer only — no torch. The model runs on the
RHOAI single-model serving platform, CPU only (m5a workers, no GPUs).

## 1. Serve the model on RHOAI

1. Data Science Project `reviewiq`, workbench with the Standard Data Science
   image (Python 3.11).
2. Run `notebooks/review.ipynb` — validates the checkpoint and exports
   `reviewiq-sentiment/1/model.onnx` (OVMS layout).
3. Upload the `reviewiq-sentiment/` folder to your S3/MinIO bucket via the
   project data connection.
4. Models → Deploy model:
   - Name: `reviewiq-sentiment`
   - Runtime: OpenVINO Model Server, framework `onnx-1`, no accelerator
   - Model path: the S3 prefix containing `1/model.onnx`
5. Wait for the InferenceService to report Loaded, note the predictor
   service/route.

## 2. Build and push the app image

```bash
podman build -t quay.io/sabhaska/reviewiq:latest .
podman push quay.io/sabhaska/reviewiq:latest
```

Or build in-cluster:

```bash
oc new-build --name reviewiq --binary --strategy docker -n reviewiq
oc start-build reviewiq --from-dir . --follow -n reviewiq
```

(then point the Deployment image at the internal registry tag)

## 3. Deploy

Check `openshift/deployment.yaml` — `INFERENCE_ENDPOINT` defaults to the
in-cluster predictor service. Adjust if your InferenceService name differs:
`oc get inferenceservice -n reviewiq` and use
`http://<name>-predictor.reviewiq.svc.cluster.local:8080`.

```bash
oc apply -k openshift/
oc get route reviewiq -n reviewiq -o jsonpath='{.spec.host}'
```

Open the route. The masthead shows live model state; use "Try sample reviews"
for a smoke test.

## Local dev

```bash
pip install -r requirements.txt
INFERENCE_ENDPOINT=https://<model-route> uvicorn app.main:app --reload --port 8080
```

## API

- `POST /api/analyze` — `{"reviews": ["...", "..."]}` → per-review stars,
  confidence, full 1–5 distribution, plus aggregate summary and themes.
- `GET /api/config` — model id and endpoint readiness.
- `GET /healthz` — liveness/readiness.

## Roadmap hooks

- Phase 2: ASIN-based lookup against McAuley-Lab/Amazon-Reviews-2023.
- Phase 3: campaign generation (listing copy, ads, email) from theme output.
