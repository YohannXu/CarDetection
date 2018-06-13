# Anaconda + CUDA9.0 + cuDNN + TensorFlow + PyTorch安装

## Anaconda

## CUDA9.0

### 卸载之前安装的CUDA

```
cd /usr/local/cuda-8.0/bin
sudo ./uninstall_cuda_8.0.pl
```

若有残余，删除cuda-8.0和cudnn
```
sudo rm -rf cuda-8.0
sudo rm -rf cudnn
```

### 安装CUDA9.0

```
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo dpkg -i cuda***.deb
sudo apt update
sudo apt install cuda
```

## cuDNN
 
千万不要下载Linux[Power8、Power9]，除非是Power8或Power9处理器

下载后解压

```
cd cuda/include/
sudo cp cudnn.h /usr/local/cuda/include/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
cd ..
cd lib64
sudo cp libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
```

## TensorFlow

```
pip install tensorflow
pip install tensorflow-gpu
```

## PyTorch

```
pip install torch torchvision
```
