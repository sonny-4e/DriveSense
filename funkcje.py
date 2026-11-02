"""
funkcje.py - parametry, przetwarzanie wstepne i funkcje pomocnicze dla main.py.
Cala matematyka HRV jest w klasie ECG (ECG.py) - tutaj tylko ja wywolujemy,
tak samo w pomiarze na zywo i w analizie offline.
"""
import numpy as np
import pandas as pd
import scipy.signal as sig
from ECG import ECG


# PARAMETRY DOMYSLNE - GUI w main.py nadpisuje je przed pomiarem / obliczeniami

PARAMETRY = {
    "fs": 250,                  # czestotliwosc probkowania [Hz] (Aidlab = 250)
    "filtr_dol": 0.5,           # filtr pasmowy Butterwortha: dolna granica [Hz]
    "filtr_gora": 40.0,         # filtr pasmowy Butterwortha: gorna granica [Hz]
    "rzad_filtru": 3,           # rzad filtra
    "MinPeakHeight": 1.5,       # detekcja R na sygnale Z-score
    "MinPeakProminence": 1.0,
    "MinPeakDistance": 0.35,    # minimalny odstep pikow [s]
    "STDscalar": 1.96,          # odrzucanie RR poza srednia +- STDscalar * odchylenie
    "min_uderzen": 20,          # okno jest 'wazne', gdy ma min. tyle uderzen...
    "max_odrzucone": 0.2,       # ...i odrzucono mniej niz taki odsetek RR
}

# nazwy cech zwracanych przez funkcje ponizej
CECHY = ["bpm", "sdnn", "rmssd", "pnn50", "sd1", "sd2", "lf", "hf", "lf_hf"]


def apply_aidlab_bandpass_filter(signal, fs=250, f_dol=0.5, f_gora=40.0, rzad=3):
    """Filtr pasmowoprzepustowy Butterwortha (filtfilt) - jak w offline_analysis.py,
    tylko z parametrami przekazywanymi z GUI."""
    nyquist = 0.5 * fs
    low = f_dol / nyquist
    high = f_gora / nyquist
    b, a = sig.butter(rzad, [low, high], btype='band')
    return sig.filtfilt(b, a, signal)


def standaryzuj(sygnal):
    """Z-score: (x - srednia) / odchylenie standardowe."""
    return (sygnal - np.mean(sygnal)) / (np.std(sygnal) + 1e-9)


def przygotuj_sygnal(surowy, p):
    """Cały preprocessing w jednym miejscu: filtr pasmowy + Z-score."""
    przefiltrowany = apply_aidlab_bandpass_filter(
        np.asarray(surowy, dtype=float), p["fs"], p["filtr_dol"], p["filtr_gora"], p["rzad_filtru"])
    return standaryzuj(przefiltrowany)


def uruchom_ecg(sygnal_std, p, nazwa="ECG"):
    """Tworzy obiekt ECG dla przygotowanego sygnalu i liczy wszystkie cechy.
    Parametry detekcji pochodza z GUI (slownik p)."""
    ecg = ECG(ecg_signal=sygnal_std, sampling_frequency=p["fs"], name=nazwa)
    ecg.init(
        offSetElim='off', deTrend='off', Reject='on',
        MinPeakHeight=p["MinPeakHeight"], MinPeakProminence=p["MinPeakProminence"],
        MinPeakDistance=p["MinPeakDistance"], STDscalar=p["STDscalar"]
    )
    return ecg


def cechy_z_ecg(ecg, p):
    """Zamienia obiekt ECG na slownik cech + ocene jakosci okna (jeden wiersz datasetu)."""
    n_beats = len(ecg.locs)
    # RR odrzucone przez variabilityReject (rrA = wszystkie, rr = po odrzuceniu)
    if len(ecg.rrA) > 0:
        odrzucone = 1 - len(ecg.rr) / len(ecg.rrA)
    else:
        odrzucone = 1.0
    ma_rr = len(ecg.rr) >= 10
    ma_roznice = ma_rr and ecg.rmsRR > 0
    ma_widmo = ecg.LHRV >= 64 and ecg.hfPower > 0

    return {
        "bpm": ecg.BPMlocal if ma_rr else np.nan,
        "sdnn": ecg.SDNN if ma_rr else np.nan,
        "rmssd": ecg.rmsRR if ma_roznice else np.nan,
        "pnn50": ecg.pNN50 if ma_roznice else np.nan,
        "sd1": ecg.SD1 if ma_roznice else np.nan,
        "sd2": ecg.SD2 if ma_roznice else np.nan,
        "lf": ecg.lfPower if ma_widmo else np.nan,
        "hf": ecg.hfPower if ma_widmo else np.nan,
        "lf_hf": ecg.lowFreqHighFreqRatio if ma_widmo else np.nan,
        "n_beats": n_beats,
        "rr_reject_frac": odrzucone,
        "amp_range": float(np.ptp(ecg.signal)) if ecg.len else 0.0,
        "valid": bool(ma_roznice and n_beats >= p["min_uderzen"] and odrzucone < p["max_odrzucone"]),
    }


def analizuj_okno_live(surowy, p, przytnij_s=1.0):
    """Analiza bufora z pomiaru na zywo (SUROWY sygnal): filtr + odciecie brzegow
    (artefakty filtfilt) + Z-score + ECG. Zwraca slownik cech."""
    sygnal = apply_aidlab_bandpass_filter(
        np.asarray(surowy, dtype=float), p["fs"], p["filtr_dol"], p["filtr_gora"], p["rzad_filtru"])
    k = int(przytnij_s * p["fs"])
    if len(sygnal) > 4 * k:
        sygnal = sygnal[k:-k]
    ecg = uruchom_ecg(standaryzuj(sygnal), p, "Live")
    return cechy_z_ecg(ecg, p)


def analizuj_nagranie(sciezka, p, baseline_s=300, okno_s=60, krok_s=15):
    """Analiza offline jednego pliku CSV (kolumna 'ECG_Value') - jak offline_analysis.py:
    filtr + Z-score calego nagrania, potem okna okno_s co krok_s.
    Etykieta: 0 = okno w calosci w baseline, 1 = okno w calosci po baseline
    (okna na granicy faz sa pomijane). Zwraca DataFrame: jedno okno = jeden wiersz,
    z kolumnami *_rel (cecha / mediana z waznych okien baseline)."""
    df = pd.read_csv(sciezka)
    if "ECG_Value" not in df.columns:
        raise ValueError("Brak kolumny 'ECG_Value' w pliku.")

    fs = p["fs"]
    sygnal_std = przygotuj_sygnal(df["ECG_Value"].to_numpy(), p)
    okno, krok, baza = int(okno_s * fs), int(krok_s * fs), int(baseline_s * fs)
    if len(sygnal_std) < okno:
        raise ValueError("Nagranie jest krotsze niz jedno okno analizy.")

    wiersze = []
    for start in range(0, len(sygnal_std) - okno + 1, krok):
        koniec = start + okno
        if koniec <= baza:
            etykieta = 0
        elif start >= baza:
            etykieta = 1
        else:
            continue
        ecg_chunk = uruchom_ecg(sygnal_std[start:koniec], p, f"Okno {start // fs}-{koniec // fs}s")
        cechy = cechy_z_ecg(ecg_chunk, p)
        cechy.update(plik=sciezka.replace("\\", "/").split("/")[-1],
                     t_start=start / fs, t_end=koniec / fs, label=etykieta)
        wiersze.append(cechy)

    wyniki = pd.DataFrame(wiersze)
    if wyniki.empty:
        raise ValueError("Brak okien do analizy (sprawdz baseline / dlugosc okna).")

    odniesienie = wyniki[(wyniki.label == 0) & wyniki.valid][CECHY].median()
    for k in CECHY:
        ref = odniesienie[k]
        wyniki[k + "_rel"] = wyniki[k] / ref if (ref and not np.isnan(ref)) else np.nan
    return wyniki