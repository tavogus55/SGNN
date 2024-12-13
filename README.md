# Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One

This repository is our implementation of 

>   Hongyuan Zhang, Yanan Zhu, and Xuelong Li,  "Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One," *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, DOI: 10.1109/TPAMI.2024.3392782, 2024.[(arXiv)](https://arxiv.org/pdf/2304.10126.pdf)[(IEEE)](https://ieeexplore.ieee.org/document/10507024)

*SGNN* attempts to further reduce the training complexity of each iteration from $\mathcal{O}(n^2) / \mathcal{O}(|\mathcal E|)$ (vanilla GNNs without acceleration tricks, e.g., [AdaGAE](https://github.com/hyzhang98/AdaGAE)) and $\mathcal O(n)$ (e.g., [AnchorGAE](https://github.com/hyzhang98/AnchorGAE-torch)) to $\mathcal O(m)$. 

Compared with other fast GNNs, SGNN can

-   (**Exact**) compute representations exactly (without sampling);
-   (**Non-linear**) use up to $L$ non-linear activations ($L$ is the number of layers);
-   (**Fast**) be trained with the real stochastic (mini-batch based) optimization algorithms. 

The comparison is summarized in the following table. 


![Comparison](figures/Comparison.jpg)



If you have issues, please email:

hyzhang98@gmail.com



## Requirements 


```
conda create --name SGNN-geometric-revised python=3.7
conda activate SGNN-geometric-revised
pip install tensorflow==2.1.0
pip install networkx==1.11
pip install scikit-learn==0.21.3
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric
pip install munkres
pip install ogb 
pip install protobuf==3.20.3
```


## How to run SGNN

>   Please ensure the data is rightly loaded

```
python run.py
python run_classfication.py
```

## How to get required data for reddit

### Classification

reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J

reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt

### Clustering

https://snap.stanford.edu/graphsage/reddit.zip

## Settings

### Node Classification

#### Cora

##### 

eta = 100, BP_count=5

```python
layers = [
    LayerParam(128, inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=60, lam=10**-3, batch_size=2708),
    LayerParam(64, inner_act=linear_func, act=relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=60, lam=10**-3, batch_size=2708),
    LayerParam(32, inner_act=linear_func, act=linear_func, gnn_type=LayerParam.EGCN,
               learning_rate=0.01, order=2, max_iter=60, lam=10**-3, batch_size=140),
]
```





#### Citeseer



eta = 100, BP_count = 3

```python
layers = [
    LayerParam(256, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=40, lam=10**-3, batch_size=1024),
    LayerParam(128, inner_act=relu_func, act=linear_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-3, order=1, max_iter=40, lam=10**-3, batch_size=140),
]
```



#### Pubmed

##### Setup

eta = 100, BP_count = 3

```python
layers = [
    LayerParam(256, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=100, lam=10**-3, batch_size=4096*2),
    LayerParam(128, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-4, order=2, max_iter=40, lam=10**-3, batch_size=2048),
]
```



### Node Clustering

| Dataset   | mask_rate | overlook_rates | layers         | max_iter | batch | BP_count | learning_rate | lam     | eta  | loss                | order | AU                        | activation             |
|-----------|-----------|----------------|----------------|----------|-------|----------|---------------|---------|------|---------------------|-------|--------------------------|------------------------|
| **Cora**  | 0.2       | None           | [128, 64, 32]  | 200      | 128   | 10       | 10^-3         | 10^-6   | 1    | loss1 / sample_size | -     | -                        | -                      |
| **Pubmed**| 0.2       | None           | [256, 128]     | 100      | 4096  | 10       | 10^-4         | 10^-6   | 10   | loss1               | 2     | relu                     | leaky_relu=5           |
| **Citeseer**| 0.2     | None           | [256, 128]     | 200      | 256   | 5        | 10^-4         | 10^-6   | 10   | loss1               | 2     | leaky_relu slope=0.2     | linear                 |
| **Reddit**| 0.2       | None           | [128, 64]      | 10000    | 512   | 5        | 10^-4         | 10^-6   | 10   | loss1               | 2     | relu                     | linear                 |

## Citation
```
@article{SGNN,
  author={Zhang, Hongyuan and Zhu, Yanan and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2024.3392782}
}
```