"""The run queue: every terminal state recorded, CPU limit enforced, resumable, drainable, code pinned."""
import io
import json
import signal

import pytest

from reach.runner.queue import Phase, Queue, read_results


def task(key, mode="ok", cpu_limit=10, cpu=0.1):
    return dict(key=key, argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", mode,
                               "--cpu", str(cpu)], cpu_limit=cpu_limit, meta=dict(mode=mode))


def run(ph, **kw):
    out = io.StringIO()
    q = Queue(ph, workers=kw.pop("workers", 4), install_signals=False, out=out, **kw)
    code = q.run()
    return q, code, out.getvalue()


@pytest.fixture
def phase(tmp_path):
    return Phase("t", runs_dir=tmp_path)


def test_every_terminal_state_is_recorded(phase):
    phase.write_manifest([task("ok"), task("tmo", "timeout", cpu_limit=1),
                          task("err", "error"), task("nores", "noresult")])
    _, code, out = run(phase)
    rows = phase.results()
    assert code == 0 and len(rows) == 4
    assert rows["ok"]["terminal_state"] == "converged" and rows["ok"]["queue_status"] == "ok"
    assert rows["tmo"]["terminal_state"] == "timeout" and rows["tmo"]["queue_status"] == "cpu_limit"
    assert rows["tmo"]["signal"] in (signal.SIGXCPU, signal.SIGKILL)
    assert rows["tmo"]["cpu_s"] >= 0.9                      # killed by CPU time, not wall time
    assert rows["err"]["terminal_state"] == "error" and rows["err"]["exit"] != 0
    assert "deliberate failure" in rows["err"]["error_tail"]
    assert rows["nores"]["terminal_state"] == "error" and rows["nores"]["exit"] == 0
    for r in rows.values():                                 # termination fields on every row
        assert {"exit", "signal", "cpu_s", "wall_s", "cpu_limit", "code_hash"} <= set(r)
    assert (phase.logs / "err.log").exists() and not (phase.logs / "ok.log").exists()
    assert "ETA" in out and "4/4" in out


def test_resume_skips_done_and_reruns_the_rest(phase):
    phase.write_manifest([task(f"k{i}") for i in range(6)])
    q1, _, _ = run(phase, limit=2)
    assert q1.session_done == 2 and len(phase.results()) == 2
    q2, _, out = run(phase)
    assert q2.n_done_before == 2 and q2.session_done == 4
    lines = phase.results_path.read_text().splitlines()
    assert len(lines) == 6 and len({json.loads(l)["key"] for l in lines}) == 6
    q3, _, out = run(phase)
    assert q3.session_done == 0 and "0 to run" in out


def test_stop_file_drains(phase):
    phase.write_manifest([task(f"k{i}") for i in range(3)])
    phase.stop_path.write_text("")
    q, _, out = run(phase)
    assert q.session_done == 0 and q.draining and "STOP file found" in out


def test_errors_rerun_only_on_request(phase):
    phase.write_manifest([task("err", "error"), task("ok")])
    run(phase)
    q, _, _ = run(phase)
    assert q.session_done == 0
    q, _, _ = run(phase, retry_errors=True)
    assert q.session_done == 1 and phase.results()["err"]["terminal_state"] == "error"


def test_code_change_is_refused(phase):
    phase.write_manifest([task("a"), task("b")])
    phase.dir.mkdir(parents=True, exist_ok=True)
    phase.results_path.write_text(json.dumps(dict(key="a", terminal_state="converged",
                                                  code_hash="000000000000")) + "\n")
    with pytest.raises(SystemExit, match="changed since earlier results"):
        run(phase)
    q, _, _ = run(phase, allow_code_change=True)
    assert q.session_done == 1


def test_tasks_import_the_code_snapshot(phase):
    phase.write_manifest([task("a")])
    q, _, _ = run(phase)
    module_file = phase.results()["a"]["module_file"]
    assert str(phase.dir / "code" / q.code_hash) in module_file
    from reach.paths import ROOT
    assert phase.results()["a"]["root"] == str(ROOT)          # not the snapshot's parent


def test_truncated_last_line_is_ignored(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(dict(key="a", terminal_state="converged")) + '\n{"key": "b", "termin')
    assert list(read_results(p)) == ["a"]


def test_manifest_rejects_duplicates_and_missing_cpu_limit(phase):
    with pytest.raises(ValueError, match="duplicate"):
        phase.write_manifest([task("a"), task("a")])
    with pytest.raises(ValueError, match="cpu_limit"):
        phase.write_manifest([dict(key="a", argv=["true"])])
