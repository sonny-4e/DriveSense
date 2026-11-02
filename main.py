"""
main.py - interfejs graficzny (PyQt5 + pyqtgraph) do:
  1) pomiaru EKG na zywo z opaski Aidlab (parametry ustawiane PRZED polaczeniem),
  2) analizy offline zapisanych plikow CSV.
Klasa ECG jest w ECG.py, funkcje pomocnicze i parametry w funkcje.py.
"""
import sys
import os
import csv
import asyncio
import threading
import queue
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QFormLayout, QGroupBox, QTabWidget, QPushButton, QLabel,
                             QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QComboBox)
from PyQt5.QtCore import QTimer, Qt
from funkcje import PARAMETRY, CECHY, analizuj_okno_live, analizuj_nagranie

# biblioteka aidlab jest potrzebna tylko do pomiaru - analiza offline dziala bez niej
try:
    from aidlab import AidlabManager, DataType, DeviceDelegate
    AIDLAB_OK = True
except ImportError:
    DeviceDelegate = object
    AIDLAB_OK = False


# POLA PARAMETROW PRZETWARZANIA WYSWIETLANE W GUI
# (klucz z funkcje.PARAMETRY, opis, min, max, krok, miejsca po przecinku)

POLA_PARAMETROW = [
    ("fs", "Próbkowanie [Hz]", 50, 1000, 10, 0),
    ("filtr_dol", "Filtr: dolna granica [Hz]", 0.05, 5, 0.1, 2),
    ("filtr_gora", "Filtr: górna granica [Hz]", 10, 120, 5, 1),
    ("rzad_filtru", "Rząd filtra", 1, 6, 1, 0),
    ("MinPeakHeight", "Min. wysokość piku R (Z-score)", 0.1, 5, 0.1, 2),
    ("MinPeakProminence", "Min. prominencja piku R", 0.1, 5, 0.1, 2),
    ("MinPeakDistance", "Min. odstęp pików [s]", 0.2, 1, 0.05, 2),
    ("STDscalar", "Odrzucanie RR: mnożnik σ", 1, 4, 0.1, 2),
    ("min_uderzen", "Min. uderzeń w oknie", 5, 200, 5, 0),
    ("max_odrzucone", "Max. odsetek odrzuconych RR", 0.05, 1, 0.05, 2),
]


def stworz_spinbox(wartosc, minimum, maksimum, krok, miejsca=2):
    """Pomocnicza: pole liczbowe z zakresem i krokiem."""
    sb = QDoubleSpinBox()
    sb.setDecimals(miejsca)
    sb.setRange(minimum, maksimum)
    sb.setSingleStep(krok)
    sb.setValue(wartosc)
    return sb


# AKWIZYCJA Z AIDLAB (wątek poboczny, jak w poprzedniej wersji)

class LiveECGAnalyzer(DeviceDelegate):
    def __init__(self, data_queue, czas_pomiaru_s, fs, sciezka):
        super().__init__()
        self.data_queue = data_queue
        self.sampling_rate = fs
        self.target_samples = czas_pomiaru_s * fs
        self.samples_collected = 0
        self.device = None
        self.sciezka = sciezka

        self.file = open(sciezka, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(["Timestamp", "ECG_Value"])

    async def did_connect(self, device):
        self.device = device
        await device.collect([DataType.ECG])

    def did_receive_ecg(self, device, timestamp, value):
        if self.samples_collected >= self.target_samples:
            return

        self.csv_writer.writerow([timestamp, value])
        self.data_queue.put(value)
        self.samples_collected += 1

        # zrzut na dysk co ok. 1 s, zeby przy przerwaniu nie stracic danych
        if self.samples_collected % self.sampling_rate == 0:
            self.file.flush()

    def zamknij(self):
        """Zamyka plik; jesli nic nie zebrano, usuwa pusty plik."""
        if not self.file.closed:
            self.file.close()
        if self.samples_collected == 0 and os.path.exists(self.sciezka):
            os.remove(self.sciezka)


def run_aidlab_thread(data_queue, stop_event, czas_pomiaru_s, fs, sciezka):
    """Do kolejki trafiaja: liczby (probki EKG), napisy (statusy) i None (koniec)."""
    async def main_async():
        analyzer = LiveECGAnalyzer(data_queue, czas_pomiaru_s, fs, sciezka)
        try:
            data_queue.put("Szukam urządzenia Aidlab...")
            manager = AidlabManager()
            devices = await manager.scan()

            if not devices:
                data_queue.put("Nie znaleziono urządzenia Aidlab!")
                return

            data_queue.put("Łączę z urządzeniem...")
            await devices[0].connect(analyzer)
            data_queue.put("Połączono - trwa pomiar.")

            while analyzer.samples_collected < analyzer.target_samples and not stop_event.is_set():
                await asyncio.sleep(0.5)

            if analyzer.device:
                await analyzer.device.disconnect()
        except Exception as e:
            data_queue.put(f"Błąd połączenia: {e}")
        finally:
            analyzer.zamknij()
            data_queue.put(None)

    asyncio.run(main_async())


# POMIAR NA ZYWO

class ZakladkaLive(QWidget):
    def __init__(self, glowne):
        super().__init__()
        self.glowne = glowne
        self.stop_event = None
        self.sciezka = ""

        # parametry pomiaru (ustawiane przed polaczeniem)
        self.sb_czas = QSpinBox(); self.sb_czas.setRange(10, 7200); self.sb_czas.setValue(120)
        self.sb_okno_wykresu = QSpinBox(); self.sb_okno_wykresu.setRange(2, 60); self.sb_okno_wykresu.setValue(10)
        self.sb_okno_analizy = QSpinBox(); self.sb_okno_analizy.setRange(10, 600); self.sb_okno_analizy.setValue(60)
        self.sb_odswiezanie = QSpinBox(); self.sb_odswiezanie.setRange(1, 60); self.sb_odswiezanie.setValue(5)
        self.le_prefix = QLineEdit("aidlab_ecg")
        self.le_folder = QLineEdit(os.getcwd())
        btn_folder = QPushButton("Wybierz...")
        btn_folder.clicked.connect(self.wybierz_folder)

        form = QFormLayout()
        form.addRow("Czas pomiaru [s]:", self.sb_czas)
        form.addRow("Okno wykresu [s]:", self.sb_okno_wykresu)
        form.addRow("Okno analizy HRV [s]:", self.sb_okno_analizy)
        form.addRow("Odświeżanie parametrów [s]:", self.sb_odswiezanie)
        form.addRow("Początek nazwy pliku:", self.le_prefix)
        wiersz_folder = QHBoxLayout()
        wiersz_folder.addWidget(self.le_folder)
        wiersz_folder.addWidget(btn_folder)
        form.addRow("Folder zapisu:", wiersz_folder)
        self.grupa_pomiaru = QGroupBox("Parametry pomiaru (przed połączeniem)")
        self.grupa_pomiaru.setLayout(form)

        self.btn_start = QPushButton("Połącz i start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        przyciski = QHBoxLayout()
        przyciski.addWidget(self.btn_start)
        przyciski.addWidget(self.btn_stop)

        self.status = QLabel("Gotowy.")
        self.info_label = QLabel("Tętno: -- | SDNN: -- | RMSSD: -- | pNN50: --")
        self.info_label.setStyleSheet("font-size: 14pt;")
        self.info_label.setAlignment(Qt.AlignCenter)

        self.wykres = pg.PlotWidget(title="Surowy sygnał EKG z opaski")
        self.wykres.setBackground('k')
        self.wykres.setLabel('bottom', 'Czas [s]')
        self.krzywa = self.wykres.plot(pen=pg.mkPen('y', width=2))

        uklad = QVBoxLayout(self)
        uklad.addWidget(self.grupa_pomiaru)
        uklad.addLayout(przyciski)
        uklad.addWidget(self.status)
        uklad.addWidget(self.info_label)
        uklad.addWidget(self.wykres, stretch=1)

        self.timer_plot = QTimer()
        self.timer_plot.timeout.connect(self.aktualizuj_wykres)
        self.timer_analysis = QTimer()
        self.timer_analysis.timeout.connect(self.aktualizuj_parametry)

    def wybierz_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder zapisu", self.le_folder.text())
        if folder:
            self.le_folder.setText(folder)

    def start(self):
        if not AIDLAB_OK:
            QMessageBox.warning(self, "Brak biblioteki", "Nie można zaimportować modułu 'aidlab'.")
            return
        p = self.glowne.pobierz_parametry()
        if p is None:
            return
        folder = self.le_folder.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder", "Folder zapisu nie istnieje.")
            return

        self.p = p
        self.kolejka = queue.Queue()
        self.stop_event = threading.Event()
        nazwa = datetime.now().strftime(f"{self.le_prefix.text()}_%Y%m%d_%H%M%S.csv")
        self.sciezka = os.path.join(folder, nazwa)

        # bufor do wykresu i dluzszy bufor do liczenia HRV (deque sam gubi najstarsze probki)
        self.bufor_wykres = deque(maxlen=self.sb_okno_wykresu.value() * p["fs"])
        self.bufor_hrv = deque(maxlen=self.sb_okno_analizy.value() * p["fs"])

        watek = threading.Thread(
            target=run_aidlab_thread,
            args=(self.kolejka, self.stop_event, self.sb_czas.value(), p["fs"], self.sciezka),
            daemon=True)
        watek.start()

        self.ustaw_stan_pomiaru(True)
        self.timer_plot.start(30)
        self.timer_analysis.start(self.sb_odswiezanie.value() * 1000)

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.btn_stop.setEnabled(False)
            self.status.setText("Zatrzymuję...")

    def ustaw_stan_pomiaru(self, trwa):
        self.btn_start.setEnabled(not trwa)
        self.btn_stop.setEnabled(trwa)
        self.grupa_pomiaru.setEnabled(not trwa)
        self.glowne.ustaw_parametry_aktywne(not trwa)

    def aktualizuj_wykres(self):
        nowe = []
        while not self.kolejka.empty():
            val = self.kolejka.get()
            if val is None:  # koniec pomiaru
                self.zakoncz_pomiar()
                return
            if isinstance(val, str):  # komunikat statusu
                self.status.setText(val)
                continue
            nowe.append(val)

        if nowe:
            self.bufor_wykres.extend(nowe)
            self.bufor_hrv.extend(nowe)
            y = np.array(self.bufor_wykres)
            y = y - np.median(y)
            x = np.arange(-len(y), 0) / self.p["fs"]
            self.krzywa.setData(x, y)

    def aktualizuj_parametry(self):
        fs = self.p["fs"]
        minimum = min(30, self.sb_okno_analizy.value()) * fs  # nie liczymy HRV z za krotkiego bufora
        if len(self.bufor_hrv) < minimum:
            self.status.setText(f"Zbieram dane do analizy HRV ({len(self.bufor_hrv) // fs}/{minimum // fs} s)...")
            return
        try:
            c = analizuj_okno_live(np.array(self.bufor_hrv), self.p)
            if np.isnan(c["bpm"]):
                self.info_label.setText("<span style='color: orange'>Nie wykryto uderzeń - sprawdź sygnał</span>")
                return
            ostrzezenie = "" if c["valid"] else " | <span style='color: orange'>słaba jakość okna</span>"
            self.info_label.setText(
                f"<span style='color: #00AA00'>Tętno: {c['bpm']:.1f} BPM</span> | "
                f"<span style='color: #0099BB'>SDNN: {c['sdnn']:.1f} ms</span> | "
                f"<span style='color: #BB8800'>RMSSD: {c['rmssd']:.1f} ms</span> | "
                f"pNN50: {c['pnn50']:.1f} %" + ostrzezenie)
        except Exception as e:
            print(f"Blad: {e}")

    def zakoncz_pomiar(self):
        self.timer_plot.stop()
        self.timer_analysis.stop()
        self.ustaw_stan_pomiaru(False)
        if os.path.exists(self.sciezka):
            self.status.setText(f"Pomiar zakończony. Zapisano: {self.sciezka}")
        else:
            self.status.setText(self.status.text() + "  (brak danych - plik nie został zapisany)")



# ANALIZA OFFLINE

KOLUMNY_TABELI = ["plik", "t_start", "label", "valid", "n_beats",
                  "bpm", "sdnn", "rmssd", "pnn50", "lf", "hf", "lf_hf"]


class ZakladkaOffline(QWidget):
    def __init__(self, glowne):
        super().__init__()
        self.glowne = glowne
        self.pliki = []
        self.wyniki = pd.DataFrame()

        # --- parametry analizy offline ---
        self.btn_pliki = QPushButton("Wybierz pliki CSV...")
        self.btn_pliki.clicked.connect(self.wybierz_pliki)
        self.lbl_pliki = QLabel("Nie wybrano plików.")
        self.sb_baseline = QSpinBox(); self.sb_baseline.setRange(0, 7200); self.sb_baseline.setValue(300)
        self.sb_okno = QSpinBox(); self.sb_okno.setRange(10, 600); self.sb_okno.setValue(60)
        self.sb_krok = QSpinBox(); self.sb_krok.setRange(1, 600); self.sb_krok.setValue(15)
        self.btn_oblicz = QPushButton("Oblicz")
        self.btn_oblicz.clicked.connect(self.oblicz)

        form = QFormLayout()
        form.addRow(self.btn_pliki, self.lbl_pliki)
        form.addRow("Koniec baseline'u (odpoczynek) [s]:", self.sb_baseline)
        form.addRow("Długość okna [s]:", self.sb_okno)
        form.addRow("Krok okna [s]:", self.sb_krok)
        grupa = QGroupBox("Parametry analizy offline")
        grupa.setLayout(form)

        self.status = QLabel("")
        self.podsumowanie = QLabel("")

        self.combo_cecha = QComboBox()
        self.combo_cecha.addItems(CECHY + [k + "_rel" for k in CECHY])
        self.combo_cecha.currentIndexChanged.connect(self.rysuj)
        wiersz_cecha = QHBoxLayout()
        wiersz_cecha.addWidget(QLabel("Cecha na wykresie:"))
        wiersz_cecha.addWidget(self.combo_cecha)
        wiersz_cecha.addStretch()

        self.wykres = pg.PlotWidget()
        self.wykres.setBackground('k')
        self.wykres.setLabel('bottom', 'Czas od początku nagrania [s]')
        self.wykres.addLegend()

        self.tabela = QTableWidget()
        self.tabela.setEditTriggers(QTableWidget.NoEditTriggers)

        uklad = QVBoxLayout(self)
        uklad.addWidget(grupa)
        uklad.addWidget(self.btn_oblicz)
        uklad.addWidget(self.status)
        uklad.addWidget(self.podsumowanie)
        uklad.addLayout(wiersz_cecha)
        uklad.addWidget(self.wykres, stretch=2)
        uklad.addWidget(self.tabela, stretch=1)

    def wybierz_pliki(self):
        pliki, _ = QFileDialog.getOpenFileNames(self, "Pliki z nagraniami", "", "CSV (*.csv)")
        if pliki:
            self.pliki = pliki
            self.lbl_pliki.setText(f"Wybrano plików: {len(pliki)}")

    def oblicz(self):
        if not self.pliki:
            QMessageBox.information(self, "Brak plików", "Najpierw wybierz pliki CSV.")
            return
        p = self.glowne.pobierz_parametry()
        if p is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        komunikaty, tabele = [], []
        try:
            for sciezka in self.pliki:
                try:
                    w = analizuj_nagranie(sciezka, p, self.sb_baseline.value(),
                                          self.sb_okno.value(), self.sb_krok.value())
                    wyjscie = os.path.join(os.path.dirname(sciezka),
                                           "features_" + os.path.basename(sciezka))
                    w.to_csv(wyjscie, index=False)  # wyniki zapisuja sie obok pliku zrodlowego
                    tabele.append(w)
                    komunikaty.append(f"{os.path.basename(sciezka)}: {len(w)} okien -> {os.path.basename(wyjscie)}")
                except Exception as e:
                    komunikaty.append(f"{os.path.basename(sciezka)}: BŁĄD - {e}")
        finally:
            QApplication.restoreOverrideCursor()

        self.status.setText("\n".join(komunikaty))
        if tabele:
            self.wyniki = pd.concat(tabele, ignore_index=True)
            self.wypelnij_tabele()
            self.pokaz_podsumowanie()
            self.rysuj()

    def pokaz_podsumowanie(self):
        """Srednie cech dla odpoczynku (0) i zadania (1) - tylko okna wazne."""
        wazne = self.wyniki[self.wyniki.valid]
        if wazne.empty:
            self.podsumowanie.setText("Brak ważnych okien - sprawdź parametry detekcji.")
            return
        srednie = wazne.groupby("label")[["bpm", "sdnn", "rmssd", "pnn50", "lf_hf"]].mean()
        nazwy = {0: "odpoczynek", 1: "zadanie"}
        linie = [f"{nazwy[l]}: BPM {r.bpm:.1f} | SDNN {r.sdnn:.1f} | RMSSD {r.rmssd:.1f} | "
                 f"pNN50 {r.pnn50:.1f} | LF/HF {r.lf_hf:.2f}" for l, r in srednie.iterrows()]
        self.podsumowanie.setText("Średnie (okna ważne)\n" + "\n".join(linie))

    def wypelnij_tabele(self):
        df = self.wyniki[KOLUMNY_TABELI]
        self.tabela.setRowCount(len(df))
        self.tabela.setColumnCount(len(KOLUMNY_TABELI))
        self.tabela.setHorizontalHeaderLabels(KOLUMNY_TABELI)
        for i, (_, wiersz) in enumerate(df.iterrows()):
            for j, wartosc in enumerate(wiersz):
                tekst = f"{wartosc:.2f}" if isinstance(wartosc, (float, np.floating)) else str(wartosc)
                self.tabela.setItem(i, j, QTableWidgetItem(tekst))
        self.tabela.resizeColumnsToContents()

    def rysuj(self):
        """Wykres wybranej cechy w czasie; kazdy plik innym kolorem, linia przerywana = koniec baseline'u."""
        self.wykres.clear()
        if self.wyniki.empty:
            return
        cecha = self.combo_cecha.currentText()
        self.wykres.setLabel('left', cecha)
        for i, (plik, grupa) in enumerate(self.wyniki[self.wyniki.valid].groupby("plik")):
            kolor = pg.intColor(i, hues=max(len(self.pliki), 1))
            self.wykres.plot(grupa["t_start"].to_numpy(), grupa[cecha].to_numpy(),
                             pen=pg.mkPen(kolor, width=2), symbol='o', symbolSize=6,
                             symbolBrush=kolor, name=plik)
        self.wykres.addItem(pg.InfiniteLine(pos=self.sb_baseline.value(), angle=90,
                                            pen=pg.mkPen('r', style=Qt.DashLine)))



# GLOWNE OKNO

class GlowneOkno(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aidlab ECG - pomiar i analiza HRV")
        self.resize(1100, 850)
        pg.setConfigOptions(antialias=True)

        # pola z parametrami przetwarzania (wartosci poczatkowe z ECG.PARAMETRY)
        self.pola = {}
        siatka = QGridLayout()
        for i, (klucz, opis, mini, maks, krok, miejsca) in enumerate(POLA_PARAMETROW):
            sb = stworz_spinbox(PARAMETRY[klucz], mini, maks, krok, miejsca)
            self.pola[klucz] = sb
            siatka.addWidget(QLabel(opis), i // 3, (i % 3) * 2)
            siatka.addWidget(sb, i // 3, (i % 3) * 2 + 1)
        self.grupa_parametrow = QGroupBox("Parametry przetwarzania EKG (wspólne dla pomiaru i analizy offline)")
        self.grupa_parametrow.setLayout(siatka)

        self.zakladki = QTabWidget()
        self.live = ZakladkaLive(self)
        self.offline = ZakladkaOffline(self)
        self.zakladki.addTab(self.live, "Pomiar na żywo")
        self.zakladki.addTab(self.offline, "Analiza offline")

        uklad = QVBoxLayout(self)
        uklad.addWidget(self.grupa_parametrow)
        uklad.addWidget(self.zakladki, stretch=1)

    def pobierz_parametry(self):
        """Zbiera parametry z GUI do slownika dla ECG.py. Zwraca None przy blednych wartosciach."""
        p = {k: sb.value() for k, sb in self.pola.items()}
        for k in ("fs", "rzad_filtru", "min_uderzen"):
            p[k] = int(p[k])
        if not (p["filtr_dol"] < p["filtr_gora"] < p["fs"] / 2):
            QMessageBox.warning(self, "Błędne parametry",
                                "Wymagane: dolna granica filtra < górna granica < fs/2.")
            return None
        return p

    def ustaw_parametry_aktywne(self, aktywne):
        self.grupa_parametrow.setEnabled(aktywne)

    def closeEvent(self, event):
        # zatrzymanie pomiaru przy zamknieciu okna, zeby plik CSV zostal poprawnie zamkniety
        if self.live.stop_event:
            self.live.stop_event.set()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    okno = GlowneOkno()
    okno.show()
    sys.exit(app.exec_())