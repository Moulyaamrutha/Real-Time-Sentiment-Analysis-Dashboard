# app.py
import os
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------------------------
# Logging setup
# ---------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("sentiment_api")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# Rotating file handler
fh = RotatingFileHandler(os.path.join(LOG_DIR, "sentiment_api.log"), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(ch_formatter)
logger.addHandler(fh)

# Uvicorn / FastAPI own logs to our logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.INFO)
uvicorn_access_logger.addHandler(ch)
uvicorn_logger = logging.getLogger("uvicorn.error")
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.addHandler(ch)

logger.info("Logger initialized")

# ---------------------------
# FastAPI app + CORS + docs
# ---------------------------
app = FastAPI(
    title="Sentiment Analysis API (BiLSTM)",
    version="1.0",
    description="Realtime sentiment analysis (BiLSTM). Single predict endpoint.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS: during dev allow all; in production lock it down
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Globals for model + tokenizer
# ---------------------------
MODEL_PATH = "sentiment_bilstm_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
LABEL_CLASSES = ['Negative', 'Neutral', 'Positive']
MAX_LEN = 100

model_lstm = None
tokenizer = None

# ---------------------------
# Text cleaning (must match training)
# ---------------------------
def ensure_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        logger.info("Downloading nltk stopwords")
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        logger.info("Downloading nltk wordnet")
        nltk.download("wordnet", quiet=True)

ensure_nltk_resources()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# ---------------------------
# Request / Response models
# ---------------------------
class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str

# ---------------------------
# Startup / Shutdown events
# ---------------------------
@app.on_event("startup")
def startup_event():
    global model_lstm, tokenizer
    t0 = time.time()
    logger.info("Starting application startup")

    # load tokenizer
    try:
        logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("Tokenizer loaded")
    except Exception as e:
        logger.exception("Failed to load tokenizer: %s", e)
        raise

    # load model
    try:
        logger.info("Loading BiLSTM model from %s", MODEL_PATH)
        # disable TF debug logs for cleaner console
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        model_lstm = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise

    logger.info("Startup finished in %.2f seconds", time.time() - t0)

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown initiated")
    # clean up TF session if needed
    try:
        tf.keras.backend.clear_session()
        logger.info("TensorFlow session cleared")
    except Exception as e:
        logger.warning("Error while clearing TF session: %s", e)

# ---------------------------
# Middleware: request logging
# ---------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s from %s", request.method, request.url.path, request.client.host if request.client else "unknown")
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled error while processing request: %s", exc)
        raise
    elapsed = (time.time() - t0) * 1000
    logger.info("Completed request: %s %s in %.2f ms (status %s)", request.method, request.url.path, elapsed, response.status_code)
    return response

# ---------------------------
# Health check
# ---------------------------
@app.get("/health", tags=["health"])
def health() -> Dict[str, str]:
    model_ready = model_lstm is not None and tokenizer is not None
    status = "ok" if model_ready else "loading"
    logger.info("Health check: %s", status)
    return {"status": status}

# ---------------------------
# Predict endpoint
# ---------------------------
@app.post("/predict", response_model=SentimentResponse, tags=["sentiment"])
def predict_sentiment(req: TextRequest):
    if model_lstm is None or tokenizer is None:
        logger.warning("Predict called before model/tokenizer ready")
        raise HTTPException(status_code=503, detail="Model is loading, try again shortly")

    raw_text = req.text or ""
    logger.info("Predict called — text length=%d", len(raw_text))

    try:
        cleaned = clean_text(raw_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        logger.debug("Input tokenized: seq_len=%d", len(seq[0]) if seq and seq[0] else 0)

        pred_probs = model_lstm.predict(pad, verbose=0)
        pred_idx = int(np.argmax(pred_probs, axis=1)[0])
        sentiment = LABEL_CLASSES[pred_idx]
        logger.info("Prediction: %s (idx=%d)", sentiment, pred_idx)
        return {"sentiment": sentiment}
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

# ---------------------------
# Error handler (optional friendly JSON)
# ---------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTPException: %s %s", exc.status_code, exc.detail)
    return fastapi.responses.JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ---------------------------
# Run (when invoked directly)
# ---------------------------
if __name__ == "__main__":
    # when running via python app.py -- use uvicorn programmatically
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
