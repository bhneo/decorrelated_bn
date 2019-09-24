import os

os.system('python ./main.py --idx=2 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=dbn       --dbn_m=16 --dbn_affine=True --batch=256 --epochs=80')
os.system('python ./main.py --idx=2 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=bn        --dbn_m=16 --dbn_affine=True --batch=256 --epochs=80')
os.system('python ./main.py --idx=2 --model=vgg --layer_num=11 --lr=0.1 --dataset=cifar10 --flip=True --crop=True --normalize=iter_norm --dbn_m=16 --iter=5 --dbn_affine=True --batch=256 --epochs=80')


