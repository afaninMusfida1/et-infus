import argparse
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fitur import NAMA_FITUR, bangun_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def buat_model(nama):
    if nama == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, random_state=42
    )


def evaluasi_loso(ds, nama):
    """Prediksi out-of-fold: tiap sesi diprediksi oleh model yang tidak pernah melihat sesi itu."""
    X, y = ds[NAMA_FITUR], ds["sisa_menit"] - ds["baseline"]   # model belajar koreksi baseline
    koreksi = np.zeros(len(ds))
    for latih, uji in LeaveOneGroupOut().split(X, y, ds["id_sesi"]):
        koreksi[uji] = buat_model(nama).fit(X.iloc[latih], y.iloc[latih]).predict(X.iloc[uji])
    return np.clip(ds["baseline"].to_numpy() + koreksi, 0, None)


def mae(ds, prediksi, mask=None):
    selisih = np.abs(prediksi - ds["sisa_menit"].to_numpy())
    return float(selisih[mask].mean() if mask is not None else selisih.mean())


def buat_grafik(ds, prediksi, nama_model, berkas):
    sesi = ds["id_sesi"].unique()[:4]
    fig, sumbu = plt.subplots(2, 2, figsize=(11, 7))
    for ax, s in zip(sumbu.ravel(), sesi):
        m = (ds["id_sesi"] == s).to_numpy()
        x = ds.loc[m, "waktu_berjalan"]
        ax.plot(x, ds.loc[m, "sisa_menit"], "k", label="aktual")
        ax.plot(x, ds.loc[m, "baseline"], "--", label="baseline")
        ax.plot(x, prediksi[m], label=nama_model)
        ax.set_title(f"Sesi {s}")
        ax.set_xlabel("waktu berjalan (menit)")
        ax.set_ylabel("sisa waktu (menit)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(berkas, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", default="data/sesi_simulasi.csv")
    ap.add_argument("--keluaran", default="model/model_eta.joblib")
    args = ap.parse_args()

    ds = bangun_dataset(pd.read_csv(args.sumber))
    akhir = (ds["fraksi_sisa"] <= 0.2).to_numpy()
    print(f"{ds['id_sesi'].nunique()} sesi, {len(ds)} baris\n")

    hasil = {"baseline": ds["baseline"].to_numpy()}
    for nama in ["ridge", "gbr"]:
        hasil[nama] = evaluasi_loso(ds, nama)

    tabel = pd.DataFrame(
        {n: {"MAE semua (menit)": mae(ds, p), "MAE 20% akhir (menit)": mae(ds, p, akhir)}
         for n, p in hasil.items()}
    ).T.round(2)
    print(tabel, "\n")

    terbaik = min(["ridge", "gbr"], key=lambda n: tabel.loc[n, "MAE semua (menit)"])
    if tabel.loc[terbaik, "MAE semua (menit)"] >= tabel.loc["baseline", "MAE semua (menit)"]:
        print("PERINGATAN: model tidak lebih baik dari baseline pada data ini.")
    print(f"Model terpilih: {terbaik}")

    buat_grafik(ds, hasil[terbaik], terbaik, "model/grafik_prediksi.png")
    model = buat_model(terbaik).fit(ds[NAMA_FITUR], ds["sisa_menit"] - ds["baseline"])
    joblib.dump(
        {
            "model": model,
            "nama_model": terbaik,
            "nama_fitur": NAMA_FITUR,
            "sumber_data": args.sumber,
            "mae_loso": tabel.loc[terbaik].to_dict(),
            "dilatih_pada": datetime.now().isoformat(timespec="seconds"),
        },
        args.keluaran,
    )
    print(f"Model disimpan ke {args.keluaran}")


if __name__ == "__main__":
    main()