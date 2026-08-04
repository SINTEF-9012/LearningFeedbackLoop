#!/usr/bin/env python3
"""Natural-playback UI demo runner.

Goal
- Make the UI demo reliable and fast to run for new users.
- Produce a real session stream (WS /streams/{session_id}) so the plot moves.
- Let inference raise alerts from the underlying signal rather than posting
    synthetic `/agent/memory/events` payloads.

What it does
1) Either POST /sessions/start-demo with a dataset-backed demo mode, or
2) POST /sessions, upload a session JSON, then POST /sessions/{sid}/start.

Run (after backend is up):
    python scripts/demo_ui_mock_run.py
    python scripts/demo_ui_mock_run.py --site_a_line2 --start-paused
    python scripts/demo_ui_mock_run.py --site_b

Then run the UI and select the printed session id.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Tiny .env reader (KEY=VALUE), enough for demo preflight."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _env_or_dotenv(dotenv: Dict[str, str], key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value is not None:
        return value
    return dotenv.get(key, default)


def _is_truthy(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_preflight_llm(
    dotenv: Dict[str, str],
    *,
    disable_llm: bool = False,
) -> tuple[bool, str]:
    """Match the backend contract for when an LLM is actually required."""

    if disable_llm:
        return False, "disabled via --disable-llm"

    generate_explanations = _is_truthy(
        _env_or_dotenv(dotenv, "GENERATE_EXPLANATIONS", "false")
    )
    if not generate_explanations:
        return False, "GENERATE_EXPLANATIONS is false"

    require_llm = _is_truthy(_env_or_dotenv(dotenv, "REQUIRE_LLM", "false"))
    if not require_llm:
        return False, "REQUIRE_LLM is false"

    return True, "REQUIRE_LLM and GENERATE_EXPLANATIONS are both true"


def _configure_runtime_llm_mode(base_url: str, *, disable_llm: bool) -> None:
    """Optionally force the backend into fallback-only mode for UI demos."""

    if not disable_llm:
        return

    try:
        resp = _patch_json(
            base_url,
            "/agent/memory/config",
            {"generate_explanations": False},
            timeout=15.0,
        )
        changed = resp.get("changed", {}) if isinstance(resp, dict) else {}
        if changed.get("generate_explanations") is False:
            print("LLM disabled for this UI demo run (generate_explanations=false).")
        else:
            print("Requested no-LLM UI demo mode.")
    except Exception as exc:
        print(f"  Warning: could not disable backend LLM explanations at runtime: {exc}")


def _ollama_preflight(*, ollama_url: str, model: str) -> None:
    base = (ollama_url or "").replace("/api/generate", "").rstrip("/")
    if not base:
        raise RuntimeError("OLLAMA_URL is empty")
    r = requests.get(f"{base}/api/tags", timeout=5.0)
    r.raise_for_status()
    tags = r.json() if r.content else {}
    models = tags.get("models") if isinstance(tags, dict) else None
    names = []
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model")
                if n:
                    names.append(str(n))
            elif isinstance(m, str):
                names.append(m)
    if model and names and model not in names:
        raise RuntimeError(f"Ollama reachable but model '{model}' not found in /api/tags")


def _warm_up_model(*, ollama_url: str, model: str) -> None:
    """Send a trivial generation request so Ollama loads the model into memory.

    Cloud/remote models (e.g. ``gpt-oss:20b-cloud``) are served externally and
    don't need a local warm-up.  If the warm-up call fails with a server error
    we check whether the model name looks like a cloud model (name contains
    ``-cloud`` or the ``/api/tags`` size field is ``0``) and fall back gracefully
    instead of aborting the whole demo.
    """
    base = ollama_url.replace("/api/generate", "").rstrip("/") if "/api/generate" in ollama_url else ollama_url.rstrip("/")
    chat_url = base + "/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": False,
        "options": {"num_predict": 5},
    }

    try:
        resp = requests.post(chat_url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        text = msg.get("content") or msg.get("thinking") or ""
        if not text.strip():
            print("  (warning: warm-up returned empty text — model may need manual check)")
    except requests.exceptions.HTTPError as exc:
        # Cloud models often return 500 on the first warm-up call because they
        # have a different lifecycle.  Check if this looks like a cloud model.
        is_cloud = _is_cloud_model(base, model)
        status = getattr(exc.response, "status_code", None)
        if is_cloud:
            print(f"  (cloud model detected — skipping warm-up; server returned {status})")
            print(f"  The model '{model}' is remote and will be called at inference time.")
        else:
            raise  # Local model warm-up failure is still fatal


def _is_cloud_model(ollama_base: str, model: str) -> bool:
    """Heuristic: a cloud model has '-cloud' in its name or reports size=0 in /api/tags."""
    if "-cloud" in model.lower():
        return True
    try:
        r = requests.get(f"{ollama_base}/api/tags", timeout=5.0)
        if r.status_code == 200:
            tags = r.json()
            for m in (tags.get("models") or []):
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or ""
                    if model.lower() in name.lower():
                        size = m.get("size", -1)
                        if size == 0 or size is None:
                            return True
    except Exception:
        pass
    return False


def _server_llm_warmup(base_url: str) -> None:
    """Call the backend's /agent/memory/llm/warmup endpoint.

    This forces the server-side LLMExplainer to re-check Ollama availability
    so the first events get real LLM descriptions.  Best-effort only — if the
    backend doesn't support the endpoint yet (older server), we just warn.
    """
    try:
        resp = requests.post(
            f"{base_url}/agent/memory/llm/warmup", json={}, timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            warmup = data.get("warmup_result", {})
            available = warmup.get("now_available", data.get("available"))
            model = data.get("model", "?")
            llm_calls = data.get("llm_call_count", "?")
            fallbacks = data.get("fallback_count", "?")
            forced = warmup.get("forced", False)
            status_text = "available" if available else "NOT available"
            extra = " (forced for cloud model)" if forced else ""
            print(
                f"Server LLM warmup: {status_text}{extra} — "
                f"model={model}, llm_calls={llm_calls}, fallbacks={fallbacks}"
            )
            if not available:
                print(
                    "  ⚠  Server reports LLM not available.  Events will use fallback text.\n"
                    "     Check: is Ollama running?  Is the model pulled?  "
                    "     Try: GET /agent/memory/llm/diagnostics for details."
                )
        elif resp.status_code == 404:
            print("  (server does not support /llm/warmup — skipping)")
        else:
            print(
                f"  ⚠  Server LLM warmup returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
    except requests.ConnectionError:
        print("  ⚠  Could not reach backend for LLM warmup — is the server running?")
    except Exception as e:
        print(f"  (LLM warmup best-effort failed: {e})")


def _base(url: str) -> str:
    return (url or "").rstrip("/")


def _post_json(base_url: str, path: str, payload: Dict[str, Any], *, timeout: float = 120.0) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _patch_json(base_url: str, path: str, payload: Dict[str, Any], *, timeout: float = 120.0) -> Dict[str, Any]:
    resp = requests.patch(f"{base_url}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _create_session(base_url: str, *, speed: float, samples_per_tick: int, start_paused: bool) -> str:
    res = _post_json(
        base_url,
        "/sessions",
        {
            "interval_ms": 100,
            "channels": None,
            "mode": "time",
            "speed": float(speed),
            "samples_per_tick": int(samples_per_tick),
            "start_paused": bool(start_paused),
        },
    )
    return str(res["session_id"])


def _upload_session_json(base_url: str, session_id: str, path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        resp = requests.post(f"{base_url}/sessions/{session_id}/upload", files={"file": f}, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def _start_session(base_url: str, session_id: str) -> Dict[str, Any]:
    return _post_json(base_url, f"/sessions/{session_id}/start", {})


def _reset_priors(base_url: str) -> None:
    try:
        _post_json(base_url, "/agent/memory/scorer/reset-priors", {})
    except Exception:
        # Best-effort only.
        pass


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _neo4j_preflight(dotenv: Dict[str, str]) -> None:
    """Verify Neo4j is reachable and the backend is configured to use it."""
    try:
        from neo4j import GraphDatabase  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit(
            "Neo4j driver not installed.  pip install neo4j>=5.0"
        )

    uri = os.environ.get("NEO4J_URI") or dotenv.get("NEO4J_URI") or "bolt://localhost:7687"
    user = os.environ.get("NEO4J_USERNAME") or dotenv.get("NEO4J_USERNAME") or "neo4j"
    pw = os.environ.get("NEO4J_PASSWORD") or dotenv.get("NEO4J_PASSWORD") or "changeme"

    print(f"Neo4j preflight: connecting to {uri} as {user}...")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            result.single()
        print("Neo4j preflight: ✓ connected")
    except Exception as e:
        raise SystemExit(
            f"Neo4j preflight FAILED — cannot connect to {uri}\n"
            f"  Start Neo4j: docker compose --profile core up -d\n"
            f"  Error: {e}"
        )
    finally:
        driver.close()

    # Verify the backend is using neo4j
    backend = os.environ.get("STORAGE_BACKEND") or dotenv.get("STORAGE_BACKEND") or "sqlite"
    if backend.lower() != "neo4j":
        print(
            f"  ⚠  STORAGE_BACKEND={backend} — the backend is NOT using Neo4j.\n"
            f"     Set STORAGE_BACKEND=neo4j in .env and restart the server.\n"
            f"     The demo will still run but the graph will be empty."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "session_json",
        nargs="?",
        default="test_data/sample_session.json",
        help="Path to a session JSON file for manual playback mode (default: test_data/sample_session.json)",
    )
    ap.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    ap.add_argument("--speed", type=float, default=0.02, help="Playback speed (1.0=real-time, 0.02=50x slower)")
    ap.add_argument("--samples-per-tick", type=int, default=8)
    ap.add_argument("--start-paused", action="store_true", help="Start playback paused so UI can attach")
    ap.add_argument("--reset-priors", action="store_true", help="Reset priors before injecting events")
    ap.add_argument(
        "--casedata", action="store_true",
        help="Use real case data events (from scripts/demo_data_casedata/) instead of synthetic ones",
    )
    ap.add_argument(
        "--labeled", action="store_true",
        help="Use labelled synthetic events (from scripts/demo_data_labeled/) with correct metrics",
    )
    ap.add_argument(
        "--site_a_line2", action="store_true",
        help="Use the server-side Site_a_line2 demo mode with a label-based seek into the pre_break region.",
    )
    ap.add_argument(
        "--site_c", action="store_true",
        help="Use the server-side SITE_C demo mode with a first-cutting-row seek.",
    )
    ap.add_argument(
        "--site_a", action="store_true",
        help="Use the server-side Site_a demo mode with a first-cutting-row seek.",
    )
    ap.add_argument(
        "--site_b", action="store_true",
        help="Use the server-side Site_b demo mode with a first-cutting-row seek.",
    )
    ap.add_argument(
        "--warmup", action="store_true",
        help="Run warmup sequence first (pre-seed memories + feedback), then continue with live playback in the same session",
    )
    ap.add_argument(
        "--neo4j", action="store_true",
        help="Enable Neo4j graph demo: preflight check, build co-occurrence graph, print Cypher exploration queries for Neo4j Browser (http://localhost:7474)",
    )
    ap.add_argument(
        "--neo4j-url", default=None,
        help="Neo4j Bolt URL (default: read from .env or bolt://localhost:7687)",
    )
    ap.add_argument(
        "--disable-llm", action="store_true",
        help="Force the UI demo to run without LLM explanations: skip Ollama preflight and disable backend generate_explanations for this run.",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dotenv = _read_dotenv(repo_root / ".env")
    base_url = _base(args.url)

    # Demo preflight: only require Ollama when the backend would actually need
    # it (REQUIRE_LLM=true and explanations enabled). This keeps the UI demo
    # usable in fallback-only mode.
    should_preflight_llm, llm_reason = _should_preflight_llm(
        dotenv,
        disable_llm=args.disable_llm,
    )
    if should_preflight_llm:
        ollama_url = _env_or_dotenv(dotenv, "OLLAMA_URL", "")
        ollama_model = _env_or_dotenv(dotenv, "OLLAMA_MODEL", "")
        print(f"LLM preflight: REQUIRE_LLM=true, checking Ollama model '{ollama_model}'...")
        try:
            _ollama_preflight(ollama_url=ollama_url, model=ollama_model)
            # Warm up the model so the first real event doesn't time out
            # waiting for Ollama to load weights into memory.
            print("LLM preflight: model listed — warming up (first call loads the model)...")
            _warm_up_model(ollama_url=ollama_url, model=ollama_model)
            print("LLM preflight: ok — model is warm")
        except Exception as e:
            raise SystemExit(
                "LLM preflight failed. Start Ollama and ensure the model exists (e.g. `ollama pull $OLLAMA_MODEL`).\n"
                f"Details: {e}"
            )
    elif args.disable_llm:
        print("LLM preflight skipped (--disable-llm). UI demo will use fallback summaries only.")
    else:
        print(f"LLM preflight skipped: {llm_reason}.")

    # ── Neo4j preflight (when --neo4j) ──
    if args.neo4j:
        _neo4j_preflight(dotenv)
        print("\nNeo4j graph demo enabled. After injection, Cypher queries will")
        print("be printed for exploring the graph at http://localhost:7474")
        if args.neo4j_url:
            os.environ["NEO4J_URI"] = args.neo4j_url

    _configure_runtime_llm_mode(base_url, disable_llm=args.disable_llm)

    # ── Warm up the *server-side* LLMExplainer ──
    # The demo-side warm-up above confirms Ollama connectivity, but the
    # backend's LLMExplainer has its own cached availability flag.  Calling
    # the server's /llm/warmup endpoint ensures the first batch of events
    # gets real LLM descriptions instead of silent fallback text.
    if not args.disable_llm and _is_truthy(_env_or_dotenv(dotenv, "GENERATE_EXPLANATIONS", "false")):
        _server_llm_warmup(base_url)

    demo_mode = None
    if args.site_a_line2:
        demo_mode = "site_a_line2"
    elif args.site_b:
        demo_mode = "site_b"
    elif args.site_a:
        demo_mode = "site_a"
    elif args.site_c or args.casedata:
        demo_mode = "site_c"

    if demo_mode is not None:
        if args.warmup:
            print("Warmup is not combined with server-side dataset demo modes; run scripts/warmup_demo_history.py separately if needed.")
        print(f"Starting server-side demo mode: {demo_mode}")
        res = _post_json(
            base_url,
            "/sessions/start-demo",
            {
                "mode": demo_mode,
                "speed": float(args.speed),
                "samples_per_tick": int(args.samples_per_tick),
                "start_paused": bool(args.start_paused),
                "reset_priors": bool(args.reset_priors),
            },
        )
        sid = str(res["session_id"])
        print(f"session_id={sid}")
        seek = res.get("seek") or {}
        if seek:
            print(f"seek={seek}")
        if args.start_paused:
            print("Playback is started but paused. Use UI Resume.")
        print("\nNow start the UI, select this session, and watch natural stream+alerts:")
        print(f"  session_id: {sid}")
        return 0

    session_path = Path(args.session_json)

    if args.labeled:
        cnc_session = Path(__file__).resolve().parent.parent / "test_data" / "cnc_session.json"
        if cnc_session.exists():
            session_path = cnc_session
            print(f"  (using CNC session data: {cnc_session.name})")
        else:
            print(f"  Warning: CNC session not found at {cnc_session}, using {session_path.name}")

    if not session_path.exists():
        raise SystemExit(f"Session file not found: {session_path}")

    warmup_sid = None
    if args.warmup:
        print("\n" + "=" * 60)
        print("Running warmup sequence (pre-seeding memories + feedback)...")
        print("=" * 60 + "\n")
        from warmup_demo_history import run_warmup

        warmup_sid = run_warmup(
            base_url,
            session_path,
            speed=float(args.speed),
            samples_per_tick=int(args.samples_per_tick),
            reset_priors=args.reset_priors,
            sleep=0.3,
        )
        print(f"\nWarmup complete. Session: {warmup_sid}")
        print("Continuing with live playback in the same session...\n")

    if args.reset_priors and not args.warmup:
        print("Resetting priors...")
        _reset_priors(base_url)

    if warmup_sid:
        sid = warmup_sid
        print(f"Reusing warmup session: {sid}")
        print("Starting playback...")
        _start_session(base_url, sid)
    else:
        print("Creating session...")
        sid = _create_session(
            base_url,
            speed=float(args.speed),
            samples_per_tick=int(args.samples_per_tick),
            start_paused=bool(args.start_paused),
        )
        print(f"session_id={sid}")

        print("Uploading session JSON...")
        up = _upload_session_json(base_url, sid, session_path)
        print(f"upload ok: channels={len(up.get('channels') or [])}")

        print("Starting playback...")
        _start_session(base_url, sid)

    if args.start_paused:
        print("Playback is started but paused. Use UI Resume.")

    print("\nNow start the UI, select this session, and watch natural stream+alerts:")
    print(f"  session_id: {sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
