# 环境要求 #
* python3
* Pytorch 1.6.0
* TensorboardX
* yaml, numpy, tqdm, imageio
* 需要gpu

# 数据集准备 #
__DIV2K数据集:__  下载后解压到load/div2k文件夹中
* [DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/）

__其他数据集:__ 下载后直接解压得到一个benchmark文件夹，将该文件夹移动到load文件夹中
* [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar）

#运行代码#
##训练##
在code文件下运行以下指令
`python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml --gpu [GPU] ` 
其中train_edsr-baseline-liif.yaml是选择运行的模型，可以到configs/train-div2k文件夹下选择其他模型，gpu选项可以选择服务器不同的gpu进行训练

##测试：##
在code文件下运行以下指令
__div2k数据集：__`bash scripts/test-div2k.sh [MODEL_PATH] [GPU]`
__其他数据集：__`bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]`
其中MODEL_PATH是模型的路径，GPU是选择服务器上的gpu序号

##用模型直接进行单张图片超分##
在code文件下运行以下指令
`python demo_time.py --input xxx.png --model [MODEL_PATH] --resolution [HEIGHT],[WIDTH] --output output.png --gpu [GPU]`
其中input后写明输入图像的路径，resolution后写明要求超分后的长和宽，output后写明输入图像的路径与名称

##使用模型测试集进行图像生成##
在code文件下运行以下指令
`python test_data.py --model[MODEL_PATH] --gpu [GPU]`
最后结果会在test_picture文件夹中生成
