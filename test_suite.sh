#!/bin/bash
echo "Running Unit Tests..."
python -m unittest tests/test_experiment_runner.py
if [ $? -eq 0 ]; then
    echo "Tests passed successfully."
else
    echo "Tests failed."
    exit 1
fi
