# Objective

Grade speakers on content of response to a spoken language question. Architecture of the grader uses prompt and response passed through a transformer style encoder, followed by a separate multi-head attention layer and finally a deep neural network. The training of the model is restricted to higher ability English speakers, as this is where content-based assessment is more relevant.

# Requirements

## Install with PyPI

pip install torch

pip install transformers
