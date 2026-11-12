"""Simulator sesi infus gravitasi.
Keluaran CSV memakai skema yang sama dengan log firmware:
id_sesi, ts, faktor_tetes, jumlah_tetes, laju_tetes, massa_g
"""
import argparse

import numpy as np
import pandas as pd

INTERVAL_DETIK = 2
MASSA_KOSONG_G = 20.0      # massa kantong/botol kosong
KERAPATAN_G_ML = 1.0       # air
JENDELA_LAJU_BARIS = 15    # laju dihitung dari 30 detik terakhir


def simulasi_satu_sesi(id_sesi, rng, faktor_tetes=20, dengan_drift=True, geser_klem=False):
    vol_awal = rng.uniform(100, 250)                              # mL
    laju_awal = rng.choice([25, 40, 60, 80]) * rng.uniform(0.9, 1.1)  # tetes/menit
    penurunan = rng.uniform(0.0, 0.35) if dengan_drift else 0.0   # laju turun saat level turun
    faktor_klem = 1.0
    derau = 0.0
    tetes_nyata = 0.0
    tetes_terhitung = 0
    ts = 0.0
    baris = []

    while tetes_nyata / faktor_tetes < vol_awal:
        vol_infus = tetes_nyata / faktor_tetes
        if geser_klem and rng.random() < INTERVAL_DETIK / 2400:   # geseran roller clamp (acak)
            faktor_klem = float(np.clip(faktor_klem * rng.uniform(0.7, 1.25), 0.4, 1.6))
        derau = 0.95 * derau + rng.normal(0, 0.02)
        laju = max(1.0, laju_awal * (1 - penurunan * vol_infus / vol_awal) * faktor_klem * (1 + derau))

        tetes_baru = int(tetes_nyata + laju / 60 * INTERVAL_DETIK) - int(tetes_nyata)
        tetes_nyata += laju / 60 * INTERVAL_DETIK
        tetes_terhitung += int(rng.binomial(tetes_baru, 0.995))   # sebagian tetes terlewat

        massa = MASSA_KOSONG_G + (vol_awal - tetes_nyata / faktor_tetes) * KERAPATAN_G_ML
        massa = max(massa, MASSA_KOSONG_G) + rng.normal(0, 0.3)
        ts += INTERVAL_DETIK
        baris.append((id_sesi, ts, faktor_tetes, tetes_terhitung, massa))

    data = pd.DataFrame(baris, columns=["id_sesi", "ts", "faktor_tetes", "jumlah_tetes", "massa_g"])
    selisih = data["jumlah_tetes"].diff(JENDELA_LAJU_BARIS)
    selisih = selisih.fillna(data["jumlah_tetes"] - data["jumlah_tetes"].iloc[0])
    lebar = np.minimum(np.arange(1, len(data) + 1), JENDELA_LAJU_BARIS) * INTERVAL_DETIK
    data["laju_tetes"] = selisih / lebar * 60
    data["massa_g"] = data["massa_g"].round(1)
    data["laju_tetes"] = data["laju_tetes"].round(2)
    return data[["id_sesi", "ts", "faktor_tetes", "jumlah_tetes", "laju_tetes", "massa_g"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumlah-sesi", type=int, default=20)
    ap.add_argument("--keluaran", default="data/sesi_simulasi.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--geser-klem", action="store_true", help="tambah geseran roller clamp acak (tak terprediksi)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    semua = [
        simulasi_satu_sesi(f"S{i + 1:02d}", rng, dengan_drift=(i % 5 != 0), geser_klem=args.geser_klem)
        for i in range(args.jumlah_sesi)
    ]
    pd.concat(semua).to_csv(args.keluaran, index=False)
    print(f"{args.jumlah_sesi} sesi disimpan ke {args.keluaran}")


if __name__ == "__main__":
    main()