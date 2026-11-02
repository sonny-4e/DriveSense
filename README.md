# DriveSense

Model uczenia maszynowego do identyfikacji poziomu stresu kierowcy w czasie rzeczywistym na podstawie danych okulograficznych i EKG (Projekt IPZ).

**Status: wczesny etap rozwoju.** Gotowy jest moduł przetwarzania EKG (pomiar na żywo z opaski Aidlab i analiza offline). Kolejne etapy: dataset, model uczący się na cechach HRV, fuzja z danymi z eye-trackingu.

## Co potrafi aplikacja

- **Pomiar na żywo** z opaski Aidlab (EKG, 250 Hz): wykres sygnału oraz bieżące parametry HRV (tętno, SDNN, RMSSD, pNN50) z oceną jakości okna.
- **Analiza offline** zapisanych nagrań CSV: podział na okna czasowe (domyślnie 60 s co 15 s), cechy HRV w dziedzinie czasu i częstotliwości, etykieta odpoczynek/zadanie oraz cechy względem własnej linii bazowej.
- **Wspólny silnik obliczeń** dla obu trybów, więc cechy na żywo i offline liczone są tak samo.
- **Parametry ustawiane w GUI** (filtr, progi detekcji załamków R, kryteria jakości) przed pomiarem i przed analizą.

## Parametry HRV

| Dziedzina | Cechy |
|---|---|
| Czas | BPM, SDNN, RMSSD, pNN50, SD1, SD2 |
| Częstotliwość | LF (0,04-0,15 Hz), HF (0,15-0,4 Hz), LF/HF |

Interpretacja parametrów i podstawy medyczne (autonomiczny układ nerwowy, HRV, ograniczenia metody) są opisane w dokumentacji w folderze `docs/`.

## Wymagania i instalacja

```
pip install -r requirements.txt
```

Moduł `aidlab` jest potrzebny tylko do pomiaru z opaski. Analiza offline działa bez niego.

## Uruchomienie

```
python main.py
```

Aplikacja ma dwie zakładki:

1. **Pomiar na żywo** – ustaw czas pomiaru, folder zapisu i parametry przetwarzania, a następnie kliknij „Połącz i start”.
2. **Analiza offline** – wskaż pliki CSV, ustaw koniec baseline'u (domyślnie 300 s) i długość okna, a następnie kliknij „Oblicz”. Wyniki zapisują się jako `features_<nazwa>.csv` obok nagrania.

## Struktura repozytorium

```
DriveSense/
├── main.py            interfejs graficzny (PyQt5) i akwizycja z Aidlab
├── funkcje.py         parametry, filtrowanie, analiza nagrań
├── ECG.py             klasa ECG: detekcja R, cechy HRV
├── docs/              dokumentacja projektu
├── matlab/            oryginalny kod MATLAB
├── data/              lokalne pomiary (ignorowane przez git)
├── requirements.txt
└── README.md
```

Pliki `main.py`, `funkcje.py` i `ECG.py` muszą leżeć w jednym folderze. Nie należy tworzyć folderu o nazwie `ECG`, bo Python wczyta go zamiast pliku `ECG.py`.

## Format danych

- **Wejście:** plik CSV z kolumnami `Timestamp` i `ECG_Value`, jedna próbka na wiersz (250 Hz).
- **Wyjście:** `features_*.csv`, jedno okno to jeden wiersz: cechy HRV, cechy względem baseline'u (`*_rel`), jakość okna (`n_beats`, `rr_reject_frac`, `valid`) oraz `t_start`, `t_end`, `label` (0 – odpoczynek, 1 – zadanie).

## Dane pomiarowe

Nagrania EKG to dane biomedyczne, dlatego nie są przechowywane w repozytorium (`data/` i pliki `aidlab_ecg_*.csv` są ignorowane przez git).

## Plan rozwoju

- [x] Przetwarzanie EKG i cechy HRV (czas i częstotliwość)
- [x] GUI: pomiar na żywo i analiza offline
- [ ] Wskaźnik jakości sygnału (SQI) i odporność na artefakty ruchowe
- [ ] Synchronizacja z danymi z eye-trackingu
- [ ] Dataset i `DataLoader` w PyTorch
- [ ] Model bazowy na cechach tabelarycznych, potem 1D-CNN
- [ ] Fuzja późna (late fusion) cech EKG i okulograficznych
