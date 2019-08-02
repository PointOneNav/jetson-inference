#!/usr/bin/env bash

set -x

install_opencv () {
        cd $HOME
        wget https://github.com/opencv/opencv/archive/3.4.2.zip?source=post_page--------------------------- -O opencv-3.4.2.zip
        unzip opencv-3.4.2.zip
        mkdir $HOME/opencv-3.4.2/build/
        cd $HOME/opencv-3.4.2/build/
        cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_PNG=OFF -D BUILD_TIFF=OFF -D BUILD_TBB=OFF -D BUILD_JPEG=OFF -D BUILD_JASPER=OFF -D BUILD_ZLIB=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_java=OFF -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=OFF -D ENABLE_NEON=ON -D WITH_OPENCL=OFF -D WITH_OPENMP=OFF -D WITH_FFMPEG=ON -D WITH_GSTREAMER=OFF -D WITH_GSTREAMER_0_10=OFF -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D WITH_GTK=ON -D WITH_VTK=OFF -D WITH_TBB=ON -D WITH_1394=OFF -D WITH_OPENEXR=OFF -D CUDA_ARCH_BIN=6.0 6.1 7.0 -D CUDA_ARCH_PTX="" -D INSTALL_C_EXAMPLES=OFF -D INSTALL_TESTS=OFF  -D BUILD_opencv_cudacodec=OFF ..
        make -j8
        sudo make install
}

setup_jetson-inference () {
        cd $HOME/nautilus/third_party/
        if [ ! -d "jetson-inference" ]; then
                git clone https://github.com/PointOneNav/jetson-inference.git
        fi

	cd jetson-inference
	git submodule update --init
	mkdir build
	cd build
	cmake ../
	make -j8
}

fetch_orbvoc () {
        cd $HOME/nautilus/third_party/ORBSLAM2/
        if [ ! -d "Vocabulary" ]; then
                mkdir ./Vocabulary/
        fi
        cd Vocabulary
        if [ ! -e "ORBvoc.txt" ]; then
                wget https://pointone-vision.s3-us-west-1.amazonaws.com/models/vocabulary/ORBvoc.txt.tar.gz
                tar xvf ORBvoc.txt.tar.gz
                rm ORBvoc.txt.tar.gz
        fi
}

setup_gpstk_agx () {
        cd $HOME/nautilus/

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
                export PATH=$PATH:/opt/local/gcc-linaro/bin
             cd ..
             rm -rf GPSTk
        cd ..
}

setup_nautilus () {
	cd $HOME
	if [ ! -d "nautilus" ]; then
		git clone --branch lucas/orb_gpu_features https://github.com/PointOneNav/nautilus.git
		echo 'export REPO_ROOT=$HOME/nautilus' >> $HOME/.bashrc
		source $HOME/.bashrc
	fi
}

setup_tensorrt () {
	cd $HOME

	lsb_release -r | grep -q '16.04'
	if [ $? == 0 ]; then
                echo "Fetching nv-tensorrt for Ubuntu 16.04..."
                wget https://pointone-public-build-assets.s3-us-west-1.amazonaws.com/nvidia/nv-tensorrt-repo-ubuntu1604-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb
                sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb
	else
		lsb_release -r | grep -q '18.04'
        	if [ $? == 0 ]; then
                	echo "Fetching nv-tensorrt for Ubuntu 18.04..."
                	wget https://pointone-public-build-assets.s3-us-west-1.amazonaws.com/nvidia/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb
                	sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb
		else
			echo "Error: Unsupported OS for nv-tensorrt"
			exit 1
		fi
	fi

        sudo apt-key add /var/nv-tensorrt-repo-cuda10.1-trt5.1.5.0-ga-20190427/7fa2af80.pub
        sudo apt-get update
        sudo apt-get install tensorrt
}

setup_gpu_orb_extractor () {
	cd $HOME/nautilus/third_party/
        if [ ! -d "gpu_orb_extractor" ]; then
                git clone https://github.com/PointOneNav/gpu_orb_extractor.git
        fi

	cd gpu_orb_extractor/
	cmake .
	make -j8
}

if [[ `uname` == 'Linux' && `uname -p` == "aarch64" ]]; then
	sudo /usr/bin/jetson_clocks    # (max out fans)
	sudo nvpmodel -q     # (query the current mode)
	sudo nvpmodel -m 0   # (enable MAX-N)

	### Add line to Crontab to max out fans at boot ###
	cd $HOME
	line="@reboot  /usr/bin/jetson_clocks"
	sudo crontab -l > mycron # write out current crontab
	echo $line >> mycron
	sudo crontab mycron # install new cron file
	rm mycron

	### Install required dependencies from apt ###
	sudo apt-get update
	sudo apt install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libglew-dev
	sudo apt-get -y install g++-5 gcc-5 default-jdk libxkbcommon-dev cmake python-pip python3-pip libboost-all-dev libwebsocketpp-dev

	### Download and build Bazel from source ###
	cd $HOME
	wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-dist.zip
	mkdir bazel-extract
	unzip bazel-0.22.0-dist.zip -d $HOME/bazel-extract/
	$HOME/bazel-extract/compile.sh
	echo 'export PATH=$HOME/bazel-extract/output:$PATH' >> $HOME/.bashrc
	source $HOME/.bashrc

	### Clone and setup Nautilus libraries ###
	setup_nautilus

	### Install 3rd-party dependency libraries
	cd $HOME/nautilus/
	./tools/setup/setup_libraries.sh # Install 3rd-party dependency libraries

	### Install opencv ###
	install_opencv

	### Setup GPSTk ###
	setup_gpstk_agx

	### Clone and setup jetson-inference repo ###
	setup_jetson-inference

	### Clone and setup gpu_orb_extractor ###
	setup_gpu_orb_extractor

	### Fetch ORBvoc.txt ###
	fetch_orbvoc

	### Setup environment ###
	cd $HOME/nautilus/third_party/jetson-inference
	./build_jetson.sh

	### Clean up ###
	rm $HOME/bazel-0.22.0-dist.zip
	rm $HOME/opencv-3.4.2.zip


elif [[ `uname` == 'Linux' && `uname -p` == "x86_64" ]]; then
        ### Install required dependencies from apt ###
        sudo apt-get update
        sudo apt install -y libwayland-dev libegl1-mesa-dev libxkbcommon-dev libdc1394-22-dev libgeographic-dev libsuitesparse-dev libjpeg-dev libturbojpeg0-dev libtiff-dev libflann-dev libglfw3-dev libglm-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libglew-dev
        sudo apt-get -y install git g++-5 g++-7 gcc-5 gcc-7 default-jdk libxkbcommon-dev cmake python-pip python3-pip libboost-all-dev cmake libwebsocketpp-dev

	### Copy gstreamer files ###
	sudo cp /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/* /usr/include/gstreamer-1.0/gst/

	### Setup appropriate nv-tensorrt library for OS ###
	setup_tensorrt
	
        ### Download and install Bazel 0.22, if not already installed ###
        bazel version | grep -q 'Build label: 0.22'
	if [ $? > 0 ]; then
		cd $HOME
		wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
		chmod +x bazel-0.22.0-installer-linux-x86_64.sh
		./bazel-0.22.0-installer-linux-x86_64.sh
		source $HOME/.bashrc
	fi

	### Clone and setup Nautilus repo if it does not exist ###
	setup_nautilus

	### Install 3rd-party dependency libraries ###
	cd $HOME/nautilus/
	./tools/setup/setup_libraries.sh

        ### Install opencv ###
	install_opencv

	### Setup GPSTk ###
	cd $HOME/nautilus/
	./tools/setup/setup_gpstk.sh

	### Clone jetson-inference if it does not exist ###
	setup_jetson-inference	
	
	### Clone gpu_orb_extractor if it does not exist ###
	setup_gpu_orb_extractor

	### Fetch ORBvoc.txt if it does not exist ###
	fetch_orbvoc

	### Clean up ###
	rm $HOME/bazel-0.22.0-installer-linux-x86_64.sh
	rm $HOME/opencv-3.4.2.zip

else
    echo "Not supported architecture"
fi

