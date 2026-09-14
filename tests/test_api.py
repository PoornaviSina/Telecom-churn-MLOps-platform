import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import home


def test_home():
    response = home()

    assert response["message"] == "Telecom Churn Prediction API is running"