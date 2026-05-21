from __future__ import annotations

import argparse
import importlib.util


def classify_status(*, fla_available: bool, triton_available: bool, gdn_import_ok: bool) -> str:
    if not fla_available:
        return "missing_fla"
    if not triton_available:
        return "missing_triton"
    if not gdn_import_ok:
        return "gdn_import_failed"
    return "ready"


def probe_imports() -> dict[str, str | bool]:
    fla_spec = importlib.util.find_spec("fla")
    triton_spec = importlib.util.find_spec("triton")
    gdn_import_ok = False
    gdn_import_error = ""

    if fla_spec is not None:
        try:
            from fla.layers import GatedDeltaNet  # noqa: F401

            gdn_import_ok = True
        except Exception as exc:
            gdn_import_error = f"{type(exc).__name__}: {exc}"

    status = classify_status(
        fla_available=fla_spec is not None,
        triton_available=triton_spec is not None,
        gdn_import_ok=gdn_import_ok,
    )
    return {
        "status": status,
        "fla_available": fla_spec is not None,
        "fla_origin": "" if fla_spec is None else str(fla_spec.origin),
        "triton_available": triton_spec is not None,
        "triton_origin": "" if triton_spec is None else str(triton_spec.origin),
        "gdn_import_ok": gdn_import_ok,
        "gdn_import_error": gdn_import_error,
    }


def print_status(status: dict[str, str | bool]) -> None:
    print(f"status={status['status']}")
    print(f"fla_available={status['fla_available']}")
    print(f"fla_origin={status['fla_origin']}")
    print(f"triton_available={status['triton_available']}")
    print(f"triton_origin={status['triton_origin']}")
    print(f"gdn_import_ok={status['gdn_import_ok']}")
    if status["gdn_import_error"]:
        print(f"gdn_import_error={status['gdn_import_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether FLA GatedDeltaNet kernels can import locally.")
    parser.add_argument("--require-ready", action="store_true", help="Exit non-zero unless FLA GDN is import-ready.")
    args = parser.parse_args()

    status = probe_imports()
    print_status(status)
    if args.require_ready and status["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
