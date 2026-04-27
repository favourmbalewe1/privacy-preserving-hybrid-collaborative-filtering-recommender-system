import subprocess
import sys
import os
import shutil


APP_NAME    = "PrivacyCF"
ENTRY_POINT = "app.py"
ICON_PATH   = None          # set to e.g. "assets/icon.ico" if you have one
OUTPUT_DIR  = "dist"
BUILD_DIR   = "build"
SPEC_DIR    = "."

# ── Data files bundled inside the exe ────────────────────────────────────────
# Format: (source_path, dest_folder_inside_bundle)
# All three land in the root of _MEIPASS so `import model_pipeline` works and
# the CSV is found by _find_dataset() in app.py.
ADDITIONAL_DATA = [
    ("final_merged_dataset.csv", "."),
    ("model_pipeline.py",        "."),
    ("recommend.py",             "."),
]

# ── Hidden imports ────────────────────────────────────────────────────────────
# PyInstaller cannot always detect dynamic / lazy imports.  List every package
# that is imported inside threads, try/except blocks, or via importlib.
HIDDEN_IMPORTS = [
    # differential privacy
    "diffprivlib",
    "diffprivlib.mechanisms",
    "diffprivlib.mechanisms.laplace",
    "diffprivlib.mechanisms.gaussian",
    "diffprivlib.utils",
    # scikit-learn
    "sklearn",
    "sklearn.svm",
    "sklearn.svm._classes",
    "sklearn.preprocessing",
    "sklearn.preprocessing._label",
    "sklearn.preprocessing._data",
    "sklearn.model_selection",
    "sklearn.model_selection._split",
    "sklearn.model_selection._validation",
    "sklearn.metrics",
    "sklearn.metrics._classification",
    "sklearn.pipeline",
    "sklearn.utils",
    "sklearn.utils._bunch",
    "sklearn.utils.multiclass",
    "sklearn.utils.validation",
    # scipy
    "scipy",
    "scipy.sparse",
    "scipy.sparse._csr",
    "scipy.spatial.distance",
    "scipy.special",
    "scipy.linalg",
    # joblib / loky backend
    "joblib",
    "joblib.externals.loky",
    "joblib.externals.loky.backend",
    "joblib.externals.loky.backend.managers",
    "joblib.externals.loky.backend.reduction",
    "joblib.externals.cloudpickle",
    # pandas / numpy
    "pandas",
    "pandas.core.dtypes.cast",
    "pandas.core.arrays.integer",
    "pandas._libs.tslibs.timedeltas",
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    # openpyxl (used by prepare_dataset via pandas read_excel)
    "openpyxl",
    "openpyxl.cell._writer",
    "openpyxl.styles",
    "openpyxl.styles.stylesheet",
    "openpyxl.reader.excel",
    # model_pipeline and recommend are shipped as .py data files but must also
    # be importable at runtime, so declare them here as well
    "model_pipeline",
    "recommend",
]

# ── Modules to strip (shrinks the exe) ───────────────────────────────────────
EXCLUDES = [
    "matplotlib",
    "IPython",
    "notebook",
    "jupyter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "gi",
    "tkinter.test",
]


# ─────────────────────────────────────────────────────────────────────────────

def install_pyinstaller():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"]
    )


def clean_build():
    for d in (OUTPUT_DIR, BUILD_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in os.listdir(SPEC_DIR):
        if f.endswith(".spec"):
            os.remove(os.path.join(SPEC_DIR, f))


def check_required_files():
    """Abort early with a clear message if any critical source file is missing."""
    missing = []
    for f in [ENTRY_POINT, "model_pipeline.py", "recommend.py"]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print("[ERROR] Required source files missing:")
        for f in missing:
            print("        " + f)
        sys.exit(1)
    if not os.path.exists("final_merged_dataset.csv"):
        print(
            "[WARN] final_merged_dataset.csv not found.\n"
            "       The exe will build and launch, but will show 'No dataset "
            "found'.\n"
            "       Run prepare_dataset.py first, then rebuild, or place the "
            "CSV next to the exe after building."
        )


def build_executable():
    install_pyinstaller()
    check_required_files()
    clean_build()

    sep = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",           # single self-contained exe
        "--windowed",          # no console window (GUI app)
        "--name",    APP_NAME,
        "--distpath", OUTPUT_DIR,
        "--workpath", BUILD_DIR,
        "--specpath", SPEC_DIR,
        "--clean",
        "--noconfirm",
    ]

    # Bundled data files
    for src, dst in ADDITIONAL_DATA:
        if os.path.exists(src):
            cmd += ["--add-data", f"{src}{sep}{dst}"]
        else:
            print(f"[SKIP] --add-data skipped (not found): {src}")

    # Hidden imports
    for hi in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", hi]

    # Excluded modules
    for ex in EXCLUDES:
        cmd += ["--exclude-module", ex]

    # Optional icon
    if ICON_PATH and os.path.exists(ICON_PATH):
        cmd += ["--icon", ICON_PATH]

    cmd.append(ENTRY_POINT)

    print("\n[BUILD] Running PyInstaller …\n")
    subprocess.check_call(cmd)

    exe_name = APP_NAME + (".exe" if sys.platform == "win32" else "")
    exe_path = os.path.join(OUTPUT_DIR, exe_name)

    if os.path.exists(exe_path):
        size_mb = round(os.path.getsize(exe_path) / 1e6, 1)
        print(f"\n[BUILD SUCCESS] {exe_path}  ({size_mb} MB)")
        print(
            "\n[NOTE] For the app to auto-load on first launch, place these "
            f"two items in the same folder as {exe_name}:\n"
            "         final_merged_dataset.csv\n"
            "         model_artifacts/hybrid_cf_model.pkl  (after training)"
        )
    else:
        print(f"[BUILD WARNING] Executable not found at: {exe_path}")
        print(f"[BUILD INFO] Contents of {OUTPUT_DIR}:")
        for f in os.listdir(OUTPUT_DIR):
            print("  " + f)


if __name__ == "__main__":
    build_executable()
