import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.logging import RichHandler

file_handler = logging.FileHandler("diabetes.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[file_handler, RichHandler(rich_tracebacks=True)],
    force=True,
)
logger = logging.getLogger(__name__)


cors_origins_raw = os.environ.get("CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
allow_credentials = False if cors_origins_raw.strip() == "*" else True
app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return JSONResponse({"status": "pass"})
