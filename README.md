This repository contains the PyTorch implemenatation of the following paper:

# Finding the relevance of sentences

```shell
python long_document_relelvance.py --model "ELECTRA" --window_length 3 --num_combinations 2

```
# Finding the fidelity of sentences


```shell
python long_document_fidelity.py --model "ELECTRA" --window_length 3 --num_combinations 2 --top_k 10 --top_explanations 4 --span_type "word"

```

```shell
python long_document_fidelity.py --model "ELECTRA" --num_combinations 2 --top_k 10 --top_explanations 4 --span_type "sent"

```
# Finding the trustworthiness of sentences

```shell
python long_document_trustworthiness.py --model "ELECTRA" --window_length 3 --num_combinations 2 --top_k 10 --top_explanations 4 --span_type "word"

```

```shell
python long_document_trustworthiness.py --model "ELECTRA" --num_combinations 2 --top_k 10 --top_explanations 4 --span_type "sent"

```
