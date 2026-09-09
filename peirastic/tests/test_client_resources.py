"""Resource setup precedes client numerical imports; no robot or live IPC."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("app,observer", [("identify_payload", True), ("gamepad", False)])
def test_client_bootstrap_runs_before_application_imports(app, observer, monkeypatch):
    from rm75_control.control.admittance_common import observer_runtime, cpu_resources
    events = []
    monkeypatch.setattr(observer_runtime, "prepare_observer_process", lambda: events.append("observer"))
    monkeypatch.setattr(observer_runtime, "limit_numeric_threads", lambda: events.append("numeric"))
    monkeypatch.setattr(cpu_resources, "prepare_background_cpus", lambda: events.append("cpus"))
    tree = ast.parse((ROOT / "peirastic/apps" / f"{app}.py").read_text())
    prefix = []
    for node in tree.body:
        prefix.append(node)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            break
    # Stop immediately after the actual top-level bootstrap; app body could
    # attach to hardware and must never be executed by this test.
    assert isinstance(prefix[-1], ast.If)
    assert not any(isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names)
                   for n in prefix)
    exec(compile(ast.Module(body=prefix, type_ignores=[]), f"{app} bootstrap", "exec"),
         {"__name__": "__main__", "__file__": str(ROOT / "peirastic/apps" / f"{app}.py")})
    assert events == (["observer"] if observer else ["numeric", "cpus"])


def test_background_exec_preserves_arguments_and_thread_affinity():
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity required")
    allowed = set(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("Need two allowed CPUs")
    excluded = min(allowed)
    from rm75_control.control.admittance_common.cpu_resources import physical_siblings
    expected = allowed - physical_siblings(excluded)
    if not expected:
        pytest.skip("No separate physical core available")
    env = dict(os.environ, RM75_OBSERVER_AVOID_CPUS=str(excluded))
    code = '''import os,sys,json,threading
data=dict(argv=sys.argv[1:],cpus=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),omp=os.environ['OMP_NUM_THREADS'],blas=os.environ['OPENBLAS_NUM_THREADS'])
def worker(): data['worker_cpus']=sorted(os.sched_getaffinity(0))
t=threading.Thread(target=worker);t.start();t.join()
print(json.dumps(data))
'''
    literal = "spaces; `literal` $(literal)"
    result = subprocess.run([sys.executable, "-m", "peirastic.apps.background", "--",
                             sys.executable, "-c", code, literal], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=15, check=True)
    data = json.loads(result.stdout)
    assert data["argv"] == [literal]
    assert set(data["cpus"]) == expected
    assert data["worker_cpus"] == data["cpus"]
    assert data["nice"] >= 10 and data["omp"] == data["blas"] == "1"


def test_background_refuses_known_control_entrypoint():
    from peirastic.apps.background import main
    with pytest.raises(SystemExit) as exc:
        main(["--", "python", "-m", "peirastic.apps.run_controller"])
    assert exc.value.code == 2
