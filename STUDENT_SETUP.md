# Getting started (for students)

This is a step-by-step guide for getting `shell_layer` running on your own
computer, written assuming little to no prior Python experience. If you
already know your way around Python and virtual environments, the short
version is in `README.md`'s "Getting started" section instead.

These instructions assume **Windows**. If you're on Mac or Linux, the
commands are the same idea but use `.venv/bin/...` instead of
`.venv\Scripts\...`.

## 1. Install Python

Skip this step if you already have Python 3.10 or newer.

1. Go to [python.org/downloads](https://www.python.org/downloads/) and
   download the latest Python 3 installer.
2. Run it. **On the very first screen, tick the checkbox "Add python.exe
   to PATH"** before clicking Install — this is the single most common
   thing people forget, and without it none of the commands below will
   work.
3. To check it worked, open a terminal (see step 3 below) and run:
   ```
   python --version
   ```
   You should see something like `Python 3.12.4`. If you instead see an
   error like "python is not recognized", Python either isn't installed
   or wasn't added to PATH — reinstall and make sure to tick that box.

## 2. Get the code

If you were given a GitHub link (e.g.
`https://github.com/christiansspathelf/shell_layer`), you have two
options — pick whichever feels easier:

- **Simplest**: on the GitHub page, click the green **Code** button →
  **Download ZIP**, then unzip it somewhere convenient (e.g. your
  Desktop or a `dev` folder).
- **If you have Git installed**: open a terminal in the folder where you
  want the project and run:
  ```
  git clone https://github.com/christiansspathelf/shell_layer.git
  ```
  This is nicer if your supervisor pushes updates later — `git pull`
  inside the folder fetches them, whereas a ZIP download needs to be
  redone from scratch each time.

## 3. Open a terminal in the project folder

- In Windows Explorer, navigate into the `shell_layer` folder (the one
  containing `pyproject.toml`, `src/`, `gui/`, etc.).
- Click the address bar, type `powershell`, and press Enter. This opens
  a PowerShell terminal already located in that folder.

## 4. Create the virtual environment

A "virtual environment" is just a private, self-contained copy of Python
for this one project, so its packages don't clash with anything else on
your machine. Create it once:

```
python -m venv .venv
```

This creates a `.venv` folder inside the project. **Do not move, rename,
or copy this folder to another machine** — it's tied to the exact Python
install and file paths on this computer. If you ever need it on another
machine, just re-run this same command there.

## 5. Install the project's dependencies into it

```
.venv\Scripts\pip install -e ".[dev,gui]"
```

This reads `pyproject.toml` and installs everything the project needs
(NumPy, pytest, Streamlit, Plotly, Matplotlib) into `.venv`, plus the
`shell_layer` package itself in "editable" mode — meaning if you or your
supervisor edit the source code afterward, you don't need to reinstall
anything to see the changes. This step needs an internet connection and
can take a minute or two the first time.

Note there's no need to "activate" the virtual environment — every
command below calls the tools inside `.venv\Scripts\` directly by their
full path, which has the same effect and is one less thing to remember.

## 6. Run the GUI

```
.venv\Scripts\streamlit run gui\app.py
```

This starts a local web server and should automatically open the app in
your browser (usually at `http://localhost:8501`). Leave the terminal
window open while you're using it — closing it stops the server. Press
`Ctrl+C` in the terminal when you're done.

## 7. (Optional) Check everything works with the test suite

```
.venv\Scripts\pytest
```

This runs the project's automated tests. Seeing everything pass (no red
`FAILED` lines) is a good sign that steps 4–5 went correctly.

## 8. (Optional) Run a plain script example

```
.venv\Scripts\python examples\basic_analysis.py
```

Useful if you want to poke at the solver directly in code rather than
through the GUI.

## If something goes wrong

- **"python is not recognized"** → Python isn't on PATH; reinstall it
  and tick "Add python.exe to PATH" (step 1).
- **`pip install` fails with a network/SSL error** → usually a firewall
  or VPN issue; try again on a different network, or ask your
  supervisor/IT.
- **PowerShell says something about "execution of scripts is disabled"**
  → this only happens if you try to *activate* the virtual environment
  (`.venv\Scripts\Activate.ps1`); since these instructions never do
  that, you shouldn't hit it. If you do, `pip`/`streamlit`/`python`
  called with their full `.venv\Scripts\...` path (as above) still work
  regardless.
- Anything else: check `README.md` (project overview and model
  description) and `CLAUDE.md` (much more detailed technical notes on
  the code itself, written for AI-assisted development but equally
  useful for understanding *why* things are built the way they are).
