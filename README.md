## Source Files

1. `train.py`  
   Trains the Potts model, saves learned model parameters to .pt.

   Minimum execution: `python src/train.py --input data/NGS-161.00.txt --model-output models/NGS-161.00.pt`

   Note on runtime: Runtime scales linearly w.r.t. data size, recommend GPU if available :) 300,000 lines (length = 10) took me 20 real minutes on CPU

2. `sampling_loop.py`  
   Generates new amino-acid sequences using batched Gibbs sampling from trained model. Writes to tsv.

   Minimum execution: `python src/sample.py --input data/NGS-161.00.txt --model models/NGS-161.00.pt --samples-output samples/NGS-161.00.tsv`

3. `eval.py`  
   Evaluates trained models and generated sequences, outputs results to tsv + txt (readable view) + duplicates file.

   Minimum execution: `python src/eval.py --input data/NGS-161.00.txt --model models/NGS-161.00.pt --samples samples/NGS-161.00.tsv --output eval_results/NGS-161.00.tsv`