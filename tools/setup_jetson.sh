#!/usr/bin/env bash

sudo /usr/bin/jetson_clocks    # (max out fans)
sudo nvpmodel -q     # (query the current mode)
sudo nvpmodel -m 0   # (enable MAX-N)

### Install required dependencies from apt ###
sudo apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libglew-dev
sudo apt-get install g++-5 gcc-5 default-jdk libxkbcommon-dev cmake python-pip python3-pip libboost-all-dev

### Download and build Bazel from source ###
cd $HOME
wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-dist.zip
mkdir bazel-extract
unzip bazel-0.22.0-dist.zip -d $HOME/bazel-extract/
$HOME/bazel-extract/compile.sh
echo 'export PATH=$HOME/bazel-extract/output:$PATH' >> $HOME/.bashrc
source $HOME/.bashrc

### Clone and setup Nautilus libraries ###
cd $HOME
git clone --branch lucas/orb_gpu_features https://github.com/PointOneNav/nautilus.git
echo 'export REPO_ROOT=$HOME/nautilus' >> $HOME/.bashrc
source $HOME/.bashrc
cd $HOME/nautilus/
$HOME/nautilus/tools/setup/setup_libraries.sh # Install 3rd-party dependency libraries

### Make and install OpenCV ###
wget https://github.com/opencv/opencv/archive/3.4.2.zip?source=post_page--------------------------- -O opencv-3.4.2.zip
unzip opencv-3.4.2.zip
mkdir $HOME/opencv-3.4.2/build/
cd $HOME/opencv-3.4.2/build/
cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_PNG=OFF -D BUILD_TIFF=OFF -D BUILD_TBB=OFF -D BUILD_JPEG=OFF -D BUILD_JASPER=OFF -D BUILD_ZLIB=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=OFF -D ENABLE_NEON=ON -D WITH_OPENCL=OFF -D WITH_OPENMP=OFF -D WITH_FFMPEG=ON -D WITH_GSTREAMER=OFF -D WITH_GSTREAMER_0_10=OFF -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D WITH_GTK=OFF -D WITH_VTK=OFF -D WITH_TBB=ON -D WITH_1394=OFF -D WITH_OPENEXR=OFF -D CUDA_ARCH_BIN=6.0 6.1 7.0 -D CUDA_ARCH_PTX="" -D INSTALL_C_EXAMPLES=OFF -D INSTALL_TESTS=OFF  -D BUILD_opencv_cudacodec=OFF ..
make -j8
sudo make install

### Setup GPSTk ###
cd $HOME/nautilus/
set -ex

if [ ! -e "WORKSPACE" ]; then
        echo "SHOULD BE RUN FROM THE ROOT OF WORKSPACE"
        exit 1
fi

cd third_party
    if [ ! -d "GPSTk" ]; then
        git clone https://github.com/PointOneNav/GPSTk.git
        cd GPSTk
    else
        cd GPSTk
        git fetch
    fi
        git checkout 35691d368f6d3669931572c9df8a90a050e19d03
        if [ ! -d "build" ]; then
            mkdir build
        fi
        cd build
            cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
            make -j8
            sudo make install
        cd ..
        rm -rf build
        mkdir build
        export PATH=$PATH:/opt/local/gcc-linaro/bin
        sudo mkdir /usr/local/aarch64-linux-gnu
        cd build
            cmake .. -DCMAKE_TOOLCHAIN_FILE=../../../toolchain_agx.cmake -DCMAKE_INSTALL_PREFIX=/usr/local/aarch64-linux-gnu
            make -j8
            sudo make install
        cd ..
        rm -rf build
    cd ..
    rm -rf GPSTk
cd ..

### Make and install websocketpp ###
cd $HOME
git clone git://github.com/zaphoyd/websocketpp.git
cd websocketpp/
cmake .
sudo make install

### Clone and setup jetson-inference repo ###
cd $HOME/nautilus/third_party/
git clone https://github.com/PointOneNav/jetson-inference.git
cd $HOME/nautilus/third_party/jetson-inference/
git submodule update --init
mkdir build
cd build
cmake ../
make -j8
sudo make install

### Clone and setup gpu_orb_extractor ###
cd $HOME/nautilus/third_party/
git clone https://github.com/PointOneNav/gpu_orb_extractor.git
cd gpu_orb_extractor/
cmake .
make -j8

### Fetch ORBvoc.txt ###
mkdir $HOME/nautilus/third_party/ORBSLAM2/Vocabulary/
cd $HOME/nautilus/third_party/ORBSLAM2/Vocabulary/
wget https://pointone-vision.s3-us-west-1.amazonaws.com/models/vocabulary/ORBvoc.txt.tar.gz
tar xvf ORBvoc.txt.tar.gz
rm ORBvoc.txt.tar.gz

### Setup environment ###
$HOME/nautilus/third_party/jetson-inference/build_jetson_armv8.sh

### Clean up ###


### Add line to Crontab to max out fans at boot ###
cd $HOME
line="@reboot  /usr/bin/jetson_clocks"
sudo crontab -l > mycron # write out current crontab
echo $line >> mycron
sudo crontab mycron # install new cron file
rm mycron

