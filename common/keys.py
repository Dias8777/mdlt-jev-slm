"""API keys come from environment variables or from a .env file. Keys are never stored in code or notebooks.

The .env file is searched in the current folder, the repo folder and their parents, so it can live
outside the repository. On Colab, put the keys into Colab "Secrets" (the key icon on the left);
get_key() reads them from there too.
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _dotenv_values():
    seen, values = set(), {}
    for start in (Path.cwd(), _REPO):
        for d in (start, *start.parents):
            f = d / ".env"
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v and k.strip() not in values:
                    values[k.strip()] = v
    return values


def get_key(name, required=True):
    value = os.environ.get(name)
    if not value:
        value = _dotenv_values().get(name)
    if not value:
        try:  # Google Colab secrets
            from google.colab import userdata  # type: ignore
            value = userdata.get(name)
        except Exception:
            value = None
    if not value and required:
        raise RuntimeError(f"{name} is not set. Put it into a .env file (see .env.example) or an environment variable.")
    return value
