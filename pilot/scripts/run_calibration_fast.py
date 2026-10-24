"""Execute an unchanged analysis with equivalent cached numerical functions.

Usage: python -I run_calibration_fast.py [--verify] analyze_ext.py <its arguments>
The verification option compares the first call of each cached function against
the original implementation on the same real inputs, failing on disagreement.
"""
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_runtime as runtime

if sys.argv[1] in ('--verify', '--verify-all'):
    runtime.VERIFY = True
    runtime.VERIFY_ALL = sys.argv[1] == '--verify-all'
    del sys.argv[1]
source = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
body = []
names = {'em', 'em_multi'}
if source.name == 'analyze_ext.py':
    names |= {'evaluate_many', 'random_label_cdf'}
for node in tree.body:
    body.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in names:
        body.extend(ast.parse(f'{node.name} = _cached.wrap({node.name}, {node.name!r})').body)
tree.body = body
namespace = {'__name__': '__main__', '__file__': str(source), '_cached': runtime}
runtime.CONTEXT = namespace
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
