"""Load the raw Home Credit inputs used by notebooks/main01.ipynb."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data(data_dir=None, *, require_history=True):
    """Return (application_train, bureau, previous_application).

    With no argument, read from the repository's data/raw folder regardless of
    the caller's working directory. ``data_dir`` may be data/ or data/raw/.
    """
    directory = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data" / "raw"
    if not directory.is_absolute():
        # Prefer a caller-relative path when it exists; otherwise resolve from repo root.
        if not directory.exists():
            directory = PROJECT_ROOT / directory
    if directory.name == "data":
        directory = directory / "raw"
    paths = {
        "application": directory / "application_train.csv",
        "bureau": directory / "bureau.csv",
        "previous": directory / "previous_application.csv",
    }
    if not paths["application"].is_file():
        raise FileNotFoundError(f"Application training data not found: {paths['application']}")

    missing_history = [paths[name] for name in ("bureau", "previous") if not paths[name].is_file()]
    if require_history and missing_history:
        raise FileNotFoundError("Required history file(s) not found: " + ", ".join(map(str, missing_history)))

    application = pd.read_csv(paths["application"])
    bureau = pd.read_csv(paths["bureau"]) if paths["bureau"].is_file() else None
    previous = pd.read_csv(paths["previous"]) if paths["previous"].is_file() else None
    print(f"Application training: {application.shape}")
    if bureau is not None:
        print(f"Bureau history: {bureau.shape}")
    if previous is not None:
        print(f"Previous applications: {previous.shape}")
    return application, bureau, previous
