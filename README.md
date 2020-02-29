# Query2box
Official implementation of [Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings](https://openreview.net/forum?id=BJgr4kSFDS).

[Hongyu Ren*](http://hyren.me), [Weihua Hu*](http://web.stanford.edu/~weihuahu/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), ICLR 2020.

## Requirements
```
torch==1.2.0
tensorboadX==1.6
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
