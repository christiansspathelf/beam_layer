# Erste Schritte (für Studierende)

Diese Anleitung erklärt Schritt für Schritt, wie du `beam_layer` auf
deinem eigenen Computer zum Laufen bringst — geschrieben für den Fall,
dass du wenig bis keine Vorerfahrung mit Python hast. Falls du dich mit
Python und virtuellen Umgebungen bereits auskennst, findest du die
Kurzversion stattdessen im Abschnitt "Getting started" der `README.md`.

Diese Anleitung geht von **Windows** aus. Auf Mac oder Linux funktionieren
die Befehle nach demselben Prinzip, verwenden aber `.venv/bin/...` statt
`.venv\Scripts\...`.

## 1. Python installieren

Diesen Schritt kannst du überspringen, falls bereits Python 3.10 oder
neuer installiert ist.

1. Gehe auf [python.org/downloads](https://www.python.org/downloads/) und
   lade den neuesten Python-3-Installer herunter.
2. Führe ihn aus. **Setze auf dem allerersten Bildschirm das Häkchen bei
   "Add python.exe to PATH"**, bevor du auf Install klickst — das ist der
   mit Abstand häufigste Punkt, der vergessen wird, und ohne ihn
   funktioniert keiner der folgenden Befehle.
3. Um zu prüfen, ob es geklappt hat, öffne ein Terminal (siehe Schritt 3
   weiter unten) und führe aus:
   ```
   python --version
   ```
   Es sollte etwas wie `Python 3.12.4` erscheinen. Falls stattdessen eine
   Fehlermeldung wie "python is not recognized" kommt, ist Python
   entweder nicht installiert oder wurde nicht zum PATH hinzugefügt —
   installiere es neu und achte diesmal auf das Häkchen.

## 2. Den Code besorgen

Falls du einen GitHub-Link erhalten hast, gibt es zwei Möglichkeiten —
wähle, was dir leichter fällt:

- **Am einfachsten**: Auf der GitHub-Seite auf den grünen Button
  **Code** → **Download ZIP** klicken und die Datei irgendwo entpacken,
  z. B. auf dem Desktop oder in einem `dev`-Ordner.
- **Falls Git installiert ist**: Öffne ein Terminal in dem Ordner, in dem
  das Projekt liegen soll, und führe aus:
  ```
  git clone <dein-repo-link>
  ```
  Das ist praktischer, falls dein Betreuer später Änderungen pusht — mit
  `git pull` im Projektordner holst du sie dann einfach nach, während ein
  ZIP-Download jedes Mal von Neuem gemacht werden müsste.

## 3. Ein Terminal im Projektordner öffnen

- Navigiere im Windows-Explorer in den Ordner `beam_layer` (den, der
  `pyproject.toml`, `src/`, `gui/` usw. enthält).
- Klicke in die Adresszeile, tippe `powershell` und drücke Enter. Damit
  öffnet sich ein PowerShell-Terminal direkt in diesem Ordner.

## 4. Die virtuelle Umgebung erstellen

Eine "virtuelle Umgebung" ist einfach eine private, in sich
abgeschlossene Kopie von Python nur für dieses eine Projekt, damit sich
seine Pakete nicht mit irgendetwas anderem auf deinem Rechner in die
Quere kommen. Einmalig erstellen mit:

```
python -m venv .venv
```

Das erzeugt einen Ordner `.venv` innerhalb des Projekts. **Diesen Ordner
nicht verschieben, umbenennen oder auf einen anderen Rechner kopieren**
— er ist an die genaue Python-Installation und die Dateipfade auf diesem
Computer gebunden. Falls du ihn auf einem anderen Rechner brauchst, führe
einfach denselben Befehl dort erneut aus.

## 5. Die Abhängigkeiten des Projekts installieren

```
.venv\Scripts\pip install -e ".[dev,gui]"
```

Dieser Befehl liest `pyproject.toml` und installiert alles, was das
Projekt benötigt (NumPy, pytest, Streamlit, Plotly, Matplotlib) in
`.venv`, zusammen mit dem `beam_layer`-Paket selbst im "editable"-Modus
— das heisst, wenn du oder dein Betreuer später den Quellcode ändern,
muss danach nichts neu installiert werden, damit die Änderungen wirksam
werden. Dieser Schritt braucht eine Internetverbindung und kann beim
ersten Mal ein bis zwei Minuten dauern.

Die virtuelle Umgebung muss dabei nicht "aktiviert" werden — jeder Befehl
weiter unten ruft die Tools in `.venv\Scripts\` direkt über ihren
vollständigen Pfad auf, was denselben Effekt hat und eine Sache weniger
ist, die man sich merken muss.

## 6. Die GUI starten

```
.venv\Scripts\streamlit run gui\app.py
```

Das startet einen lokalen Webserver und sollte die App automatisch im
Browser öffnen (normalerweise unter `http://localhost:8501`). Lasse das
Terminal-Fenster offen, solange du die App benutzt — schliesst du es,
stoppt der Server. Mit `Ctrl+C` im Terminal beendest du ihn, wenn du
fertig bist.

## 7. (Optional) Mit der Testsuite prüfen, ob alles funktioniert

```
.venv\Scripts\pytest
```

Das führt die automatisierten Tests des Projekts aus. Wenn alles ohne
rote `FAILED`-Zeilen durchläuft, ist das ein gutes Zeichen, dass Schritt
4–5 korrekt geklappt haben.

## Falls etwas nicht funktioniert

- **"python is not recognized"** → Python ist nicht im PATH; neu
  installieren und dabei "Add python.exe to PATH" ankreuzen (Schritt 1).
- **`pip install` schlägt mit einem Netzwerk-/SSL-Fehler fehl** → meist
  ein Firewall- oder VPN-Problem; in einem anderen Netzwerk nochmal
  versuchen, oder Betreuer/IT fragen.
- **PowerShell meldet etwas über "execution of scripts is disabled"**
  → das passiert nur, wenn man versucht, die virtuelle Umgebung zu
  *aktivieren* (`.venv\Scripts\Activate.ps1`); da diese Anleitung das nie
  tut, sollte das nicht auftreten. Falls doch: `pip`/`streamlit`/`python`
  funktionieren trotzdem, wenn sie wie oben über ihren vollständigen
  `.venv\Scripts\...`-Pfad aufgerufen werden.
- Bei allem anderen: `README.md` (Projektüberblick und Modellbeschreibung)
  und `CLAUDE.md` (deutlich detailliertere technische Hinweise zum Code
  selbst, geschrieben für KI-gestützte Entwicklung, aber genauso
  hilfreich, um zu verstehen, *warum* die Dinge so gebaut sind, wie sie
  sind) konsultieren.
