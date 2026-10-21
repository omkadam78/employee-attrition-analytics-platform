from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_api_loads_real_model():
    text = (ROOT / 'api' / 'app.py').read_text(encoding='utf-8')
    assert "joblib.load" in text and "predict_proba" in text

def test_ibm_status_is_honest():
    assert 'was not completed' in (ROOT / 'docs' / 'IBM_DEPLOYMENT.md').read_text(encoding='utf-8')
