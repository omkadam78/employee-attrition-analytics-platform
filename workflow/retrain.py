from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'src' / 'train.py'), run_name='__main__')
print('Retraining completed. Review reports/results before promotion.')
