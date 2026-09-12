"""Execute the staged supervisor done branch without importing hardware code."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('status,message,error,completed',[
    (2,'icra_scan:tracking',0,True),
    (1,'icra_wait',0,True),
    (1,'icra_scan:tracking',0,False),
    (3,'confidence feedback stale',1,False),
    (2,'contact_qp_final_review_rejected',1,False),
])
def test_termination_is_not_success(status,message,error,completed):
    source=Path(__file__).with_name('scan_robot.py.staged').read_text()
    tree=ast.parse(source)
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and 'done_seq' in ast.unparse(n.test) and 'without confirmed completion' in ast.unparse(n))
    marks=[]
    env=dict(snap={'done_seq':7,'status':status,'err_code':error},message=message,
             self=NS(expected_seq=7,mark=marks.append),tracking_started=10.,Status=NS(DONE=2))
    module=ast.Module(body=[ast.For(target=ast.Name(id='_once',ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(1)],ctx=ast.Load()),body=[branch],orelse=[])],type_ignores=[])
    code=compile(ast.fix_missing_locations(module),'<staged supervisor completion>','exec')
    if completed:
        exec(code,env)
        assert marks==['path_done']
    else:
        with pytest.raises(RuntimeError,match=message):exec(code,env)
        assert marks==[]
