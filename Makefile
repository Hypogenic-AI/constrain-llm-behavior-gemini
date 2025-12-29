.PHONY: install verify test run plot analyze inspect demo all clean

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

verify:
	python check_data.py

analyze:
	python analyze_dataset.py

test:
	python -m unittest tests/test_experiment_runner.py

run:
	python experiment_runner.py --model_name gpt2 --num_samples 50 --num_generations 3

plot:
	python plot_results.py

inspect:
	python inspect_results.py

demo:
	python mock_demo.py

all: install verify test run plot

clean:
	rm -rf results/*
	rm -rf __pycache__
