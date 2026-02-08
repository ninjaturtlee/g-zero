import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # last resort
    return str(x)


def write_audit_bundle(bundle: Dict[str, Any], out_dir: str | Path = "artifacts/audit") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _safe_git_commit()
    bundle = {
        "timestamp_utc": _utc_now_iso(),
        "git_commit": git_commit,
        "cwd": os.getcwd(),
        "bundle": _to_jsonable(bundle),
    }

    fname = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path = out_dir / fname
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    return path