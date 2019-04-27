# gym-summarizer
The Summarizer environment is a domain for text generation in the context of summarization. 
In the future, these tasks will be supported:

## Extractive Summarization
Generate summaries by extracting sentence from reference documents.

## How to replicate experiments
1. Download [CNN-DailyMail finished file binaries](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail);  
2. Use gym_summarizer.utils.BatchCNNDMLoader to precompute embeddings (WARNING: precomputing will take a very long time and the resulting files will occupy >30Gb of storage);
3. Use run_experiment.py and evaluate_model.py to train agents and evaluate them.
