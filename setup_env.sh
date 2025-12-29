#!/bin/bash
echo "Setting up environment..."
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "Setup complete."
