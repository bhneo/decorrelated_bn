import os

os.system('python ./main.py --idx=1 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=dbn --dbn_m=16 --dbn_affine=True --batch=128 --epochs=80')
# os.system('python ./main.py --idx=1 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=bn --dbn_m=16 --dbn_affine=True --batch=256 --epochs=80')
os.system('python ./main.py --idx=1 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=iter_norm --iter=5 --dbn_m=16 --dbn_affine=True --batch=128 --epochs=80')

