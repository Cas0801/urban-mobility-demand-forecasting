PYTHON ?= python

.PHONY: install data prepare train backtest test dashboard api monitor pipeline container

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.download --year 2024 --months 1 2 3

prepare:
	$(PYTHON) -m src.prepare

train:
	$(PYTHON) -m src.train

backtest:
	$(PYTHON) -m src.backtest

test:
	$(PYTHON) -m pytest -q

dashboard:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn src.api:app --reload

monitor:
	$(PYTHON) -m src.monitor

pipeline: data prepare train backtest test

container:
	docker build -t urban-mobility-forecasting .
	docker run --rm -p 8501:8501 urban-mobility-forecasting
