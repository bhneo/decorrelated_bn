# Decorrelated Batch Normalization
An implementation of DecorrelatedBN by tensorflow


This implementation is based on papers: 
<b>Decorrelated Batch Normalization</b> (https://arxiv.org/abs/1804.08450) 
from Lei Huang, Dawei Yang, Bo Lang, Jia Deng.

### Running experiments
```buildoutcfg
python vgg.py --type=A --batch_size=256 --lr=0.1 --method=dbn --m=0
```
model_name += '_{}'.format(params.model.type)
    model_name += '_bs{}'.format(params.training.batch_size)
    model_name += '_lr{}'.format(params.training.lr)
    model_name += '_{}'.format(params.normalize.method)
    model_name += '_m{}'.format(params.normalize.m)
    if params.normalize.method == 'iter_norm':
        model_name += '_iter{}'.format(params.normalize.iter)
    if params.normalize.affine:
        model_name += '_affine'
    model_name += '_idx{}'.format(str(params.training.idx))

recent result on cnn:

<img src="result/cnn.jpg"></img>



### Reference
More deteils please refer to the torch edition by Huang Lei 
- [umich-vl/DecorrelatedBN](https://github.com/umich-vl/DecorrelatedBN)


