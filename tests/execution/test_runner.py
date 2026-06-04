"""execution.runner — HALT, ChainMode dispatch, heartbeat checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from shared_core.contracts import ChainMode

from execution.alpaca_client import AlpacaClient
from execution.journal import read_fills
from execution.runner import execute_batch


# ---------------------------------------------------------------------------
# ChainMode dispatch
# ---------------------------------------------------------------------------


def test_whatif_is_pure_no_broker_no_journal(
    make_intent, fake_alpaca, journal_root: Path
) -> None:
    """WHATIF must not submit and must not touch the filesystem."""

    intent = make_intent(intent_id="intent-WI")
    # Note: we still pass a client to prove it's NOT called.
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.WHATIF, client=client, journal_root=journal_root
    )
    assert len(result.outcomes) == 1
    out = result.outcomes[0]
    assert out.status == "WHATIF"
    assert out.client_order_id is not None
    assert len(fake_alpaca.submitted) == 0
    # No parquet files should have been written.
    assert list(journal_root.rglob("*.parquet")) == []


def test_paper_submits_and_writes_journal(
    make_intent, fake_alpaca, journal_root: Path
) -> None:
    intent = make_intent(intent_id="intent-PAPER", qty="4")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.PAPER, client=client, journal_root=journal_root
    )
    assert result.submitted == 1
    assert len(fake_alpaca.submitted) == 1
    parquet_files = list(journal_root.rglob("*.parquet"))
    assert len(parquet_files) == 1
    table = read_fills(parquet_files)
    row = table.to_pylist()[0]
    assert row["mode"] == "paper"
    assert row["status"] == "SUBMITTED"
    assert row["intent_id"] == "intent-PAPER"


def test_live_submits_and_writes_journal(
    make_intent, fake_alpaca, journal_root: Path
) -> None:
    """LIVE goes through the same code path on the Alpaca-only setup."""

    intent = make_intent(intent_id="intent-LIVE")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.LIVE, client=client, journal_root=journal_root
    )
    assert result.submitted == 1
    parquet_files = list(journal_root.rglob("*.parquet"))
    assert len(parquet_files) == 1
    table = read_fills(parquet_files)
    assert table.to_pylist()[0]["mode"] == "live"


# ---------------------------------------------------------------------------
# HALT gating
# ---------------------------------------------------------------------------


def test_halt_blocks_paper_submission(
    make_intent,
    fake_alpaca,
    journal_root: Path,
    isolate_state_dir: Path,
) -> None:
    (isolate_state_dir / "HALT").touch()
    intent = make_intent(intent_id="intent-HALT")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.PAPER, client=client, journal_root=journal_root
    )
    out = result.outcomes[0]
    assert out.status.startswith("REFUSED")
    assert "HALT" in out.status
    # Broker not called; no journal write either.
    assert fake_alpaca.submitted == []
    assert list(journal_root.rglob("*.parquet")) == []


def test_halt_does_not_block_whatif(
    make_intent,
    fake_alpaca,
    journal_root: Path,
    isolate_state_dir: Path,
) -> None:
    (isolate_state_dir / "HALT").touch()
    intent = make_intent(intent_id="intent-WI-HALT")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.WHATIF, client=client, journal_root=journal_root
    )
    assert result.outcomes[0].status == "WHATIF"


# ---------------------------------------------------------------------------
# Heartbeat gating
# ---------------------------------------------------------------------------


def test_stale_heartbeat_blocks(
    make_intent,
    fake_alpaca,
    journal_root: Path,
    isolate_state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-enable the heartbeat freshness gate (default conftest disables it).
    monkeypatch.delenv("MACMINIBOT_DISABLE_HEARTBEAT_CHECK", raising=False)
    # Write a heartbeat 10 minutes old → stale (limit is 5 min).
    p = isolate_state_dir / "heartbeat.json"
    p.write_text(json.dumps({"pid": 1, "ts_unix": time.time() - 600, "version": "x"}))

    intent = make_intent(intent_id="intent-STALE")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.PAPER, client=client, journal_root=journal_root
    )
    out = result.outcomes[0]
    assert "HEARTBEAT_STALE" in out.status


def test_fresh_heartbeat_allows_submission(
    make_intent,
    fake_alpaca,
    journal_root: Path,
    write_fresh_heartbeat,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MACMINIBOT_DISABLE_HEARTBEAT_CHECK", raising=False)
    write_fresh_heartbeat(age_s=1.0)
    intent = make_intent(intent_id="intent-FRESH")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [intent], mode=ChainMode.PAPER, client=client, journal_root=journal_root
    )
    assert result.submitted == 1


# ---------------------------------------------------------------------------
# Multi-intent batch — HALT mid-batch must block subsequent orders
# ---------------------------------------------------------------------------


def test_halt_midbatch_blocks_subsequent_orders(
    make_intent,
    fake_alpaca,
    journal_root: Path,
    isolate_state_dir: Path,
) -> None:
    """Per Contract 2: HALT is rechecked per-order, never cached."""

    intent_a = make_intent(intent_id="intent-A")
    intent_b = make_intent(intent_id="intent-B", ticker="TSLA")

    # Wrap submit_order so HALT trips between order A and order B.
    original_submit = fake_alpaca.submit_order
    halt_file = isolate_state_dir / "HALT"

    def submit_then_halt(order_data):  # noqa: ANN001
        reply = original_submit(order_data)
        halt_file.touch()
        return reply

    fake_alpaca.submit_order = submit_then_halt  # type: ignore[assignment]
    client = AlpacaClient(client=fake_alpaca)

    result = execute_batch(
        [intent_a, intent_b],
        mode=ChainMode.PAPER,
        client=client,
        journal_root=journal_root,
    )
    statuses = [o.status for o in result.outcomes]
    assert statuses[0] == "SUBMITTED"
    assert statuses[1].startswith("REFUSED")
    assert len(fake_alpaca.submitted) == 1


# ---------------------------------------------------------------------------
# Connection / mode validation
# ---------------------------------------------------------------------------


def test_execute_batch_rejects_non_chainmode(make_intent) -> None:
    with pytest.raises(TypeError):
        execute_batch([make_intent()], mode="paper")  # type: ignore[arg-type]


def test_execute_batch_requires_client_for_live(make_intent, journal_root) -> None:
    with pytest.raises(RuntimeError, match="AlpacaClient is required"):
        execute_batch(
            [make_intent()], mode=ChainMode.PAPER, client=None, journal_root=journal_root
        )


def test_broker_exception_is_recorded_as_rejected(
    make_intent, fake_alpaca, journal_root: Path
) -> None:
    fake_alpaca.raise_on_submit = RuntimeError("connection refused")
    client = AlpacaClient(client=fake_alpaca)
    result = execute_batch(
        [make_intent(intent_id="intent-REJ")],
        mode=ChainMode.PAPER,
        client=client,
        journal_root=journal_root,
    )
    out = result.outcomes[0]
    assert out.status == "REJECTED"
    assert "connection refused" in (out.reason or "")
    # The rejected row should still be journaled for audit.
    parquet_files = list(journal_root.rglob("*.parquet"))
    assert len(parquet_files) == 1
    row = read_fills(parquet_files).to_pylist()[0]
    assert row["status"] == "REJECTED"
