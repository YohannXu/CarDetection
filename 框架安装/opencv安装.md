# 1.编译安装ffmpeg

## 1.1安装依赖

```
sudo apt install yasm libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libxext-dev libxfixes-dev
```

## 1.2编译

```
./configure --prefix=/usr/local/ffmpeg --enable-shared --enable-gpl --enable-swscale
sudo make
sudo make install
```

## 1.3测试

```
ffmpeg -version
```

# 2.编译安装opencv

## 2.1安装依赖

## 2.2编译

```
cd opencv-版本号
mkdir release
cd release
cmake -D WITH_IPP=OFF 
      -D PYTHON_DEFAULT_EXECUTABLE=/home/yohannxu/anaconda3/bin/python3
      -D BUILD_opencv_python3=ON
      -D BUILD_opencv_python2=OFF
      -D PYTHON3_EXECUTABLE=/home/yohannxu/ananconda3/bin/python3
      -D PYTHON3_INCLUDE_DIR=/home/yohannxu/anaconda3/include/python3.6m
      -D PYTHON3_LIBRARY=/home/yohannxu/anaconda3/lib/libpython3.6m.so.1.0
      -D NUMPY_PATH=/home/yohannxu/anaconda3/lib/site-packages
      -D CUDA_GENERATOR=Maxwell (看显卡版本更改)
      ..
sudo make -j56 (数字为cpu核心)
sudo make install
cp /home/yohannxu/下载/opencv-3.4.1/release/lib/python3/XXX.so /home/yohannxu/anaconda3/lib/python3.6/site-packages/
```
