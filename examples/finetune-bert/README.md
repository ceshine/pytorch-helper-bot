# Fine-tuning BERT Examples

1. [01-BERT-sst2.ipynb](./01-BERT-sst2.ipynb): Fine-tuning BERT-base-uncased on SST-2 using `huggingface/transformers` and `huggingface/nlp`.
   - Taken from [ceshine/transformer_to_rnn](https://github.com/ceshine/transformer_to_rnn/tree/20200616-blog-post).
   - Reference: [[Failure Report] Distill Fine-tuned Transformers into Recurrent Neural Networks](https://blog.ceshine.net/post/failed-to-distill-transformer-into-rnn/).
   - PyTorch-Helper-Bot version: 0.6.0
2. [02-BERT-sst2-DeepSpeed.py](./02-BERT-sst2-DeepSpeed.py): Experimental DeepSpeed support
   - BERT-base-uncased:
     - Train with FP16: `deepspeed 02-BERT-sst2-DeepSpeed.py --arch bert-base-uncased --config gpu.json` (This runs slightly faster than 01-BERT-sst2, which uses APEX amp O1.)
     - Train with FP16 + ZeRO-Offload: `deepspeed 02-BERT-sst2-DeepSpeed.py --arch bert-base-uncased --config cpu.json 1>/dev/null` (This runs much slower but allows you to use bigger batch sizes.)
   - BERT-large-uncased
     - Train with FP16: `deepspeed 02-BERT-sst2-DeepSpeed.py --arch bert-large-uncased --config gpu.json` (This will get a OOM if you're using a 8GB graphic card)
     - Train with FP16 + ZeRO-Offload: `deepspeed 02-BERT-sst2-DeepSpeed.py --arch bert-large-uncased --config cpu.json 1>/dev/null` (This will run but also use 24+GB RAM on your computer)
