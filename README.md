# Decorrelated Batch Normalization
This project provides the Tensorflow implementation of ZCA whitening described in paper:  

**[Decorrelated Batch Normalization](https://arxiv.org/abs/1804.08450)**(CVPR 2018)

and IterNorm whitening in paper:

**[Iterative Normalization: Beyond Standardization towards Efficient Whitening](https://arxiv.org/abs/1904.03441)**(CVPR 2019)

### Requirements
* python3
* seaborn
* matplotlib
* easydict
* tensorflow >= 2.0.0


### Running experiments
To reproduce the VGG-network experiment, just run "vgg.py" and pass the config parameters.
For example: 
```buildoutcfg
python vgg.py --type=A --batch=256 --lr=0.1 --method=zca --m=0
```
where the "type" denotes the type of VGG-network architecture, 
"batch" denotes the batch size, "lr" denotes the initial learning rate,
"method" denotes the whitening method (zca, iter_norm), 
"m" denotes the group size (0 indicates full whitening).

### Usage


### Reference
More deteils please refer to the implementations:
- Torch: [princeton-vl/DecorrelatedBN](https://github.com/princeton-vl/DecorrelatedBN)
- Pytorch: [huangleiBuaa/IterNorm-pytorch](https://github.com/huangleiBuaa/IterNorm-pytorch)