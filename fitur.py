import pandas as pd

INTERVAL_DETIK = 2          # interval log firmware, harus tetap
MASSA_KOSONG_G = 20.0       # massa kantong/botol kosong (ukur, lalu sesuaikan)
KERAPATAN_G_ML = 1.0        # air
JENDELA_RATA = 15           # 30 detik
JENDELA_KEMIRINGAN = 30     # 60 detik
JENDELA_MASSA = 5           # 10 detik
BASELINE_MAKS_MENIT = 600
MIN_BARIS = JENDELA_RATA + JENDELA_KEMIRINGAN

NAMA_FITUR = [
    "laju_tetes", "laju_rata", "kemiringan_laju",
    "vol_infus", "vol_sisa", "waktu_berjalan", "baseline",
    "fraksi_terinfus", "proyeksi_perubahan",
]


def hitung_fitur(data, faktor_tetes, ts_mulai):
    """Fitur per baris, hanya memakai data sampai baris itu (tanpa kebocoran)."""
    f = pd.DataFrame(index=data.index)
    f["laju_tetes"] = data["laju_tetes"]
    f["laju_rata"] = data["laju_tetes"].rolling(JENDELA_RATA, min_periods=JENDELA_RATA).mean()
    f["kemiringan_laju"] = f["laju_rata"].diff(JENDELA_KEMIRINGAN) / (JENDELA_KEMIRINGAN * INTERVAL_DETIK / 60)
    f["vol_infus"] = data["jumlah_tetes"] / faktor_tetes
    massa_halus = data["massa_g"].rolling(JENDELA_MASSA, min_periods=JENDELA_MASSA).mean()
    f["vol_sisa"] = ((massa_halus - MASSA_KOSONG_G) / KERAPATAN_G_ML).clip(lower=0)
    f["waktu_berjalan"] = (data["ts"] - ts_mulai) / 60
    laju_ml_menit = (f["laju_rata"] / faktor_tetes).clip(lower=0.05)
    f["baseline"] = (f["vol_sisa"] / laju_ml_menit).clip(upper=BASELINE_MAKS_MENIT)
    total = f["vol_infus"] + f["vol_sisa"]
    f["fraksi_terinfus"] = f["vol_infus"] / total.clip(lower=1e-6)
    kemiringan_relatif = f["kemiringan_laju"] / f["laju_rata"].clip(lower=1)   # perubahan laju per menit
    f["proyeksi_perubahan"] = kemiringan_relatif * f["baseline"]              # perubahan laju selama sisa waktu
    return f


def bangun_dataset(data):
    """Fitur + label untuk seluruh sesi. Label = menit sampai baris terakhir sesi (kantong kosong)."""
    bagian = []
    for id_sesi, g in data.groupby("id_sesi"):
        g = g.sort_values("ts").reset_index(drop=True)
        ts_mulai, ts_kosong = g["ts"].iloc[0], g["ts"].iloc[-1]
        f = hitung_fitur(g, g["faktor_tetes"].iloc[0], ts_mulai)
        f["sisa_menit"] = (ts_kosong - g["ts"]) / 60
        f["fraksi_sisa"] = f["sisa_menit"] / ((ts_kosong - ts_mulai) / 60)
        f["id_sesi"] = id_sesi
        f = f.dropna()
        bagian.append(f[f["laju_rata"] >= 1])
    return pd.concat(bagian, ignore_index=True)