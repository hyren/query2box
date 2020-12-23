# Query2box

**A new and more modularized implementation can be found at [here](https://github.com/snap-stanford/KGReasoning)**

Official implementation of [Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings](https://openreview.net/forum?id=BJgr4kSFDS).

[Hongyu Ren*](http://hyren.me), [Weihua Hu*](http://web.stanford.edu/~weihuahu/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), ICLR 2020.

## Installation

To install in development mode, clone from GitHub
with the following:

```bash
git clone https://github.com/hyren/query2box
cd query2box
pip install --editable .
```

`--editable` means that the code is symlinked into your
Python's `site-packages` so it doesn't need to be reinstalled
every time the code is changed.

## Command Line Interface

The `query2box` command line interface is installed automatically. It can be used
like in the following:

```bash
$ CUDA_VISIBLE_DEVICES=0 query2box --do_train --cuda --do_valid --do_test \
    --data_path data/FB15k --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
    -lr 0.0001 --max_steps 300000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
    --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc --stepsforpath 300000  --offset_deepsets inductive \
    --center_deepsets eleattention --print_on_screen
```

## Run
To reproduce the results on FB15k, FB15k-237 and NELL, the hyperparameters are set in `example.sh`.
```
bash example.sh
```

## Cite
Please consider cite our paper if you find the paper and the code useful.
```
@inproceedings{ren2020query2box,
  title={Query2box: Reasoning over Knowledge Graphs in Vector Space using Box Embeddings},
  author={Ren, Hongyu and Hu, Weihua and Leskovec, Jure},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=BJgr4kSFDS},
}
```

Feel free to send email to hyren@cs.stanford.edu if you have any questions. This code also borrows from [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
