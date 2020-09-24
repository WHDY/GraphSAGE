## GraphSAGE

GraphSAGE of Pytorch version. 

GCN-mean, mean, pooling and Attention based aggregated methods are implemented. 

### Usage

- Mean aggregator

  ```
  python train.py --agg-type Mean --out-dim 128,128
  ```

- GCN-Mean aggregator

  ```
  python train.py --agg-type GCN --out-dim 128,128
  ```

- Pooling aggregator(max)

  ```p
  python train.py --agg-type Pooling --pool-fun max --h-dim 128
  ```

- Attention aggregator

  ```
  python train.py --agg-type Attention --heads 3 --out-dim 384,128 --concat 1,0
  ```

