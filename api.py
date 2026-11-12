import hmac
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from fitur import MIN_BARIS, hitung_fitur

BERKAS_MODEL = os.getenv("BERKAS_MODEL", "model/model_eta.joblib")
KUNCI_API = os.getenv("KUNCI_API", "")


@asynccontextmanager
async def siklus_hidup(app: FastAPI):
    app.state.bundel = joblib.load(BERKAS_MODEL)
    yield


app = FastAPI(title="API Prediksi Sisa Waktu Infus", lifespan=siklus_hidup)


def cek_kunci(x_api_key: str = Header(default="")):
    if KUNCI_API and not hmac.compare_digest(x_api_key, KUNCI_API):
        raise HTTPException(status_code=401, detail="Kunci API salah")


class Pembacaan(BaseModel):
    ts: float
    jumlah_tetes: int = Field(ge=0)
    laju_tetes: float = Field(ge=0)
    massa_g: float


class PermintaanPrediksi(BaseModel):
    faktor_tetes: float = Field(gt=0)
    ts_mulai: float
    pembacaan: list[Pembacaan]


class ResponsPrediksi(BaseModel):
    status: str                      # ok | belum_cukup_data | aliran_berhenti
    eta_menit: float | None = None
    eta_baseline_menit: float | None = None
    sumber: str | None = None        # model | baseline


@app.get("/kesehatan")
def kesehatan(request: Request):
    b = request.app.state.bundel
    return {
        "status": "ok",
        "model": b["nama_model"],
        "sumber_data": b["sumber_data"],
        "mae_loso": b["mae_loso"],
        "dilatih_pada": b["dilatih_pada"],
    }


@app.post("/prediksi", response_model=ResponsPrediksi, dependencies=[Depends(cek_kunci)])
def prediksi(req: PermintaanPrediksi, request: Request):
    bundel = request.app.state.bundel
    if len(req.pembacaan) < MIN_BARIS:
        return ResponsPrediksi(status="belum_cukup_data")

    data = pd.DataFrame([p.model_dump() for p in req.pembacaan]).sort_values("ts").reset_index(drop=True)
    fitur = hitung_fitur(data, req.faktor_tetes, req.ts_mulai).iloc[[-1]][bundel["nama_fitur"]]
    if fitur.isna().any().any():
        return ResponsPrediksi(status="belum_cukup_data")
    if fitur["laju_rata"].iloc[0] < 1:
        return ResponsPrediksi(status="aliran_berhenti")

    baseline = float(fitur["baseline"].iloc[0])
    eta = baseline + float(bundel["model"].predict(fitur)[0])
    sumber = "model"
    if not np.isfinite(eta) or eta > 3 * baseline + 5:   # keluaran tidak masuk akal -> baseline
        eta, sumber = baseline, "baseline"
    return ResponsPrediksi(
        status="ok",
        eta_menit=round(max(eta, 0.0), 1),
        eta_baseline_menit=round(baseline, 1),
        sumber=sumber,
    )