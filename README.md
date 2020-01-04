<img src="https://pbs.twimg.com/profile_images/830189200041316353/6Hb9DajI_400x400.jpg">


## Point one setup instructions:

### Cloning:

```
cd ~/nautilus/third_party/
git clone https://github.com/pointonenav/jetson-inference/
```

### Dependencies setup:

Prerequisites:
```
sudo apt install libgstreamer-plugins-base1.0-dev
sudo apt install libgstreamer1.0-dev
sudo apt install libglew-dev
```

Installation for tensorRT on Ubuntu x86_64:

Download the appropriate library for your OS:

Ubuntu 16.04

https://pointone-public-build-assets.s3-us-west-1.amazonaws.com/nvidia/nv-tensorrt-repo-ubuntu1604-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb

Ubuntu 18.04

https://pointone-public-build-assets.s3-us-west-1.amazonaws.com/nvidia/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb 
OR
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb 

sudo apt-key add /var/nv-tensorrt-repo-cuda10.1-trt5.1.5.0-ga-20190427/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```


For more info:
https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-install-guide/index.html#installing-debian


If tensorrt fails try installing this:
```
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
```
Next, re-run:
```
sudo apt-get install tensorrt
sudo apt-get install cuda-10-1
```

### Pre-emptive build error fixing:

```
sudo ln -s /usr/lib/x86_64-linux-gnu/glib-2.0/include/glibconfig.h /usr/include/glib-2.0/
```


### Building:

```
cd jetson-inference
git submodule update --init
sudo apt-get install libqt4-dev
cmake .
make -j4
```

### Downloading models:

```
cd data
./download_model.sh
```

# Nvidia AGX Xavier Setup

There are two options to choose from to setup the required software on AGX Xavier for CV:

1) The setup script `setup_orb.sh`, which is located in the tools directory of this repo. The downside of this method is that it can take a very long time for cloning multiple AGXs, as all of the downloading and compilation must be done locally on the AGX.

2) The Xavier cloning procedure, which is described below. While it does take significant time to make a backup of the "master" AGX, clones can flashed very quickly compared with the above method.

# Xavier Cloning Procedure

### Prerequisites

The host machine should be running [Ubuntu 16.04 64-bit](http://releases.ubuntu.com/16.04/) and have at least 56 GB of free disk space.

There must be one "master" AGX Xavier, with any desired configurations and compiled files on it. The "master" will be backed up to a network drive, which will then be used to make clones. The master must be on the same network as the host machine.

### NFS configuration on host

Install NFS on the host machine with the following command:

`sudo apt-get install nfs-kernel-server`

Install the [Nvidia Jetpack SDK](https://developer.nvidia.com/embedded/jetpack) for AGX Xavier

Make a directory that [NFS](https://en.wikipedia.org/wiki/Network_File_System) will share. We will name the directory `nfs-share` and place it at the root of the filesystem, but the name and location are arbitrary and long as they remain consistent.

`sudo mkdir /nfs-share` 

#### Configuring NFS Exports

Open `/etc/exports` in a text editor, and modify it as follows. Here, using vim:

`sudo vim /etc/exports`

Add the following line to make the directory accessible by all IP addresses on the network:

`/nfs-share       *(rw,sync,no_root_squash,no_subtree_check)`

Alternatively, if security is a concern, `/etc/exports` can be modified to only give read and write privileges to certain IP addresses. In our case, the IP address of each AGX Xavier must be known and each should be added on a new line as follows (with the example IP addresses 192.168.1.100 and 192.168.1.200):

```
/nfs-share       192.168.1.100(rw,sync,no_root_squash,no_subtree_check)
/nfs-share       192.168.1.200(rw,sync,no_root_squash,no_subtree_check)
```

When all changes are done being made to `/etc/exports`, issue the following command:

`sudo systemctl restart nfs-kernel-server`

This will implement the new NFS rules.

#### Firewalls

Make sure there are no firewalls enabled on the host by issuing the following command:

`sudo ufw status`

If ufw is installed, this command should return `Status: inactive`, indicating there are no active firewalls. Otherwise, disable ufw:

`sudo ufw disable`

### NFS configuration on master AGX Xavier

All of the commands in this section should be executed on the "master" AGX Xavier.

Issue the following to install client-side NFS:

`sudo apt-get install nfs-common`

#### Make a directory to mount to the NFS

`sudo mkdir /nfs`

#### Create mount point

Create a mount point on the Xavier. Replace `host_ip` with the host's IP address on the network. 

`sudo mount host_ip:/nfs-share /nfs`

#### Checking for successful configuration

Run `df -h` and there should be a line resembling the following:

```
Filesystem                          				        Size  Used Avail Use% Mounted on
host_ip:/nfs-share 							XXXG   XXG  XXXG   X% /nfs
```

In addition, you should be able to create a test file in the `/nfs` directory and have that file appear on the host.

`sudo touch /nfs/testfile`

If `testfile` appears on the host in `/nfs-share/` then the configuration was successful. Delete this file after a successful configuration is confirmed.

### Cloning master AGX Xavier to NFS drive

Using rsync, we clone the master AGX Xavier to the network drive (`/nfs-share` on the host) 

`sudo rsync -aAXv / --exclude={"/dev/*","/nfs/*","/proc/*","/sys/*","/tmp/*","/run/*","/mnt/*","/media/*","/lost+found"} /nfs/` 

The command-line arguments `-aAXv` will ensure that all symbolic links, devices, permissions, ownerships, modification times, ACLs, and extended attributes are preserved. Importantly, we must exclude the directory `/nfs`, or else rsync will run in an infinite loop, since this is the directory we are writing the AGX's filesystem to.

This process may take a long time.

## Making the backup

### Booting off the NFS

To boot off the NFS, put the master AGX Xavier into recovery mode. See the [Jetson AGX Xavier User Guide](https://developer.download.nvidia.com/embedded/L4T/r32-2_Release_v1.0/jetson_agx_xavier_developer_kit_user_guide.pdf) for instructions on how to do this. Once the AGX is in recovery mode and plugged into the host machine via the front USB-C port on the AGX, issue the following commands on the host machine:

```
cd $JETPACK_ROOT/JetPack_4.2_Linux_P2888/Linux_for_Tegra/
sudo ./flash.sh -N <ip-addr-of-linux-host>:/nfs-share --rcm-boot jetson-xavier eth0
```

This will make the master AGX boot off of the network. 

### Generate the image

Now that the AGX is booted off of the AGX and not the eMMC, the rootfs is the NFS. Therefore, the actual eMMC is frozen, so we can run the following commands to generate a backup image:

```
sudo umount /dev/mmcblk0p1
sudo dd if=/dev/mmcblk0p1 of=/rootfs.img
```

The dd command may take a long time to execute, since it is creating an image of the entire root filesystem.  Since we are booted off the NFS, when the dd command is finished, rootfs.img should appear as `/nfs-share/rootfs.img` on the host.

Verify that rootfs.img exists on the host, then shutdown the master AGX.

### Generate sparse image

On the host, issue the following:

```
cd $JETPACK_ROOT/JetPack_4.2_Linux_P2888/Linux_for_Tegra/bootloader/
sudo ./mksparse  -v --fillpattern=0 /nfs-share/rootfs.img system.img
```

## Flashing Clones

Place the AGX Xavier you would like to clone into recovery mode, and plug it into the host via the front USB-C port on the AGX. Then, issue the following:

```
cd $JETPACK_ROOT/JetPack_4.2_Linux_P2888/Linux_for_Tegra/
sudo ./flash.sh -r jetson-xavier mmcblk0p1
```

This will flash the AGX Xavier, creating a clone of the master AGX.

Repeat the instructions in this section to make as many clones as desired.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Deploying Deep Learning
Welcome to our training guide for inference and [deep vision](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html) runtime library for NVIDIA **[DIGITS](https://github.com/NVIDIA/DIGITS)** and **[Jetson Xavier/TX1/TX2](http://www.nvidia.com/object/embedded-systems.html)**.

This repo uses NVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)** for efficiently deploying neural networks onto the embedded platform, improving performance and power efficiency using graph optimizations, kernel fusion, and half-precision FP16 on the Jetson.

Vision primitives, such as [`imageNet`](imageNet.h) for image recognition, [`detectNet`](detectNet.h) for object localization, and [`segNet`](segNet.h) for semantic segmentation, inherit from the shared [`tensorNet`](tensorNet.h) object.  Examples are provided for streaming from live camera feed and processing images from disk.  See the **[Deep Vision API Reference Specification](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html)** for accompanying documentation. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-primitives.png" width="800">

There are multiple tracks of the tutorial that you can choose to follow, including Training + Inference or Inference-Only.

> &gt; &nbsp; Jetson Nano Developer Kit and JetPack 4.2 is now supported in the repo. <br/>
> &gt; &nbsp; See our technical blog including benchmarks, [`Jetson Nano Brings AI Computing to Everyone`](https://devblogs.nvidia.com/jetson-nano-ai-computing/).

## Hello AI World (Inference Only)

If you would like to only do the inference portion of the tutorial, which can be run on your Jetson in roughly two hours, these modules are available below:

* [Setting up Jetson with JetPack](docs/jetpack-setup-2.md)
* [Building the Repo from Source](docs/building-repo-2.md)
* [Classifying Images with ImageNet](docs/imagenet-console-2.md)
	* [Using the Console Program on Jetson](docs/imagenet-console-2.md#using-the-console-program-on-jetson)
	* [Coding Your Own Image Recognition Program](docs/imagenet-example-2.md)
	* [Running the Live Camera Recognition Demo](docs/imagenet-camera-2.md)
* [Locating Object Coordinates using DetectNet](docs/detectnet-console-2.md)
	* [Detecting Objects from the Command Line](docs/detectnet-console-2.md#detecting-objects-from-the-command-line)
	* [Running the Live Camera Detection Demo](docs/detectnet-camera-2.md)

## Two Days to a Demo (Training + Inference)

The full tutorial includes training and inference, and can take roughly two days or more depending on system setup, downloading the datasets, and the training speed of your GPU.

* [DIGITS Workflow](docs/digits-workflow.md) 
* [DIGITS System Setup](docs/digits-setup.md)
* [Setting up Jetson with JetPack](docs/jetpack-setup.md)
* [Building the Repo from Source](docs/building-repo.md)
* [Classifying Images with ImageNet](docs/imagenet-console.md)
	* [Using the Console Program on Jetson](docs/imagenet-console.md#using-the-console-program-on-jetson)
	* [Coding Your Own Image Recognition Program](docs/imagenet-example.md)
	* [Running the Live Camera Recognition Demo](docs/imagenet-camera.md)
	* [Re-Training the Network with DIGITS](docs/imagenet-training.md)
	* [Downloading Image Recognition Dataset](docs/imagenet-training.md#downloading-image-recognition-dataset)
	* [Customizing the Object Classes](docs/imagenet-training.md#customizing-the-object-classes)
	* [Importing Classification Dataset into DIGITS](docs/imagenet-training.md#importing-classification-dataset-into-digits)
	* [Creating Image Classification Model with DIGITS](docs/imagenet-training.md#creating-image-classification-model-with-digits)
	* [Testing Classification Model in DIGITS](docs/imagenet-training.md#testing-classification-model-in-digits)
	* [Downloading Model Snapshot to Jetson](docs/imagenet-snapshot.md)
	* [Loading Custom Models on Jetson](docs/imagenet-custom.md)
* [Locating Object Coordinates using DetectNet](docs/detectnet-training.md)
	* [Detection Data Formatting in DIGITS](docs/detectnet-training.md#detection-data-formatting-in-digits)
	* [Downloading the Detection Dataset](docs/detectnet-training.md#downloading-the-detection-dataset)
	* [Importing the Detection Dataset into DIGITS](docs/detectnet-training.md#importing-the-detection-dataset-into-digits)
	* [Creating DetectNet Model with DIGITS](docs/detectnet-training.md#creating-detectnet-model-with-digits)
	* [Testing DetectNet Model Inference in DIGITS](docs/detectnet-training.md#testing-detectnet-model-inference-in-digits)
	* [Downloading the Detection Model to Jetson](docs/detectnet-snapshot.md)
	* [DetectNet Patches for TensorRT](docs/detectnet-snapshot.md#detectnet-patches-for-tensorrt)
	* [Detecting Objects from the Command Line](docs/detectnet-console.md)
	* [Multi-class Object Detection Models](docs/detectnet-console.md#multi-class-object-detection-models)
	* [Running the Live Camera Detection Demo on Jetson](docs/detectnet-camera.md)
* [Semantic Segmentation with SegNet](docs/segnet-dataset.md)
	* [Downloading Aerial Drone Dataset](docs/segnet-dataset.md#downloading-aerial-drone-dataset)
	* [Importing the Aerial Dataset into DIGITS](docs/segnet-dataset.md#importing-the-aerial-dataset-into-digits)
	* [Generating Pretrained FCN-Alexnet](docs/segnet-pretrained.md)
	* [Training FCN-Alexnet with DIGITS](docs/segnet-training.md)
	* [Testing Inference Model in DIGITS](docs/segnet-training.md#testing-inference-model-in-digits)
	* [FCN-Alexnet Patches for TensorRT](docs/segnet-patches.md)
	* [Running Segmentation Models on Jetson](docs/segnet-console.md)

## Extra Resources

In this area, links and resources for deep learning developers are listed:

* [Appendix](docs/aux-contents.md)
	* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes
     * [NVIDIA AI IoT](https://github.com/NVIDIA-AI-IOT) - NVIDIA Jetson GitHub repositories
     * [Jetson eLinux Wiki](https://www.eLinux.org/Jetson) - Jetson eLinux Wiki

## Recommended System Requirements

Training GPU:  Maxwell, Pascal, Volta, or Turing-based GPU (ideally with at least 6GB video memory)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;optionally, AWS P2/P3 instance or Microsoft Azure N-series  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 14.04 x86_64 or Ubuntu 16.04 x86_64.

Deployment:    &nbsp;&nbsp;Jetson Xavier Developer Kit with JetPack 4.0 or newer (Ubuntu 18.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX2 Developer Kit with JetPack 3.0 or newer (Ubuntu 16.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX1 Developer Kit with JetPack 2.3 or newer (Ubuntu 16.04 aarch64).

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against the following BSP versions for Jetson AGX Xavier and Jetson TX1/TX2: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson Nano - JetPack 4.2 / L4T R32.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.2 / L4T R32.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.1.1 DP / L4T R31.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 GA<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.1 DP EA / L4T R31.0.2 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 RC<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.0 DP EA / L4T R31.0.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 RC<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 4.2 / L4T R32.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.3 / L4T R28.2.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 4.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.3 / L4T R28.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 4.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.2 / L4T R28.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 2.1<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 2.1<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 1.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 1.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)

Note that TensorRT samples from the repo are intended for deployment onboard Jetson, however when cuDNN and TensorRT have been installed on the host side, the TensorRT samples in the repo can be compiled for PC.


## Legacy Links

<details open>
<summary>Since the documentation has been re-organized, below are links mapping the previous content to the new locations.</summary>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(click on the arrow above to hide this section)

### DIGITS Workflow

See [DIGITS Workflow](docs/digits-workflow.md)

### System Setup

See [DIGITS Setup](docs/digits-setup.md)

#### Running JetPack on the Host

See [JetPack Setup](docs/jetpack-setup.md)

#### Installing Ubuntu on the Host

See [DIGITS Setup](docs/digits-setup.md#installing-ubuntu-on-the-host)

#### Setting up host training PC with NGC container	

See [DIGITS Setup](docs/digits-setup.md#setting-up-host-training-pc-with-ngc-container)

#### Installing the NVIDIA driver

See [DIGITS Setup](docs/digits-setup.md#installing-the-nvidia-driver)

#### Installing Docker

See [DIGITS Setup](docs/digits-setup.md#installing-docker)

#### NGC Sign-up 

See [DIGITS Setup](docs/digits-setup.md#ngc-sign-up)

#### Setting up data and job directories

See [DIGITS Setup](docs/digits-setup.md#setting-up-data-and-job-directories)

#### Starting DIGITS container

See [DIGITS Setup](docs/digits-setup.md#starting-digits-container)

#### Natively setting up DIGITS on the Host 

See [DIGITS Native Setup](docs/digits-native.md)

#### Installing NVIDIA Driver on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-nvidia-driver-on-the-host)

#### Installing cuDNN on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-cudnn-on-the-host)

#### Installing NVcaffe on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-nvcaffe-on-the-host)

#### Installing DIGITS on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-digits-on-the-host)

#### Starting the DIGITS Server

See [DIGITS Native Setup](docs/digits-native.md#starting-the-digits-server)

### Building from Source on Jetson

See [Building the Repo from Source](docs/building-repo.md)
      
#### Cloning the Repo

See [Building the Repo from Source](docs/building-repo.md#cloning-the-repo)

#### Configuring with CMake

See [Building the Repo from Source](docs/building-repo.md#configuring-with-cmake)

#### Compiling the Project

See [Building the Repo from Source](docs/building-repo.md#compiling-the-project)

#### Digging Into the Code

See [Building the Repo from Source](docs/building-repo.md#digging-into-the-code)

### Classifying Images with ImageNet

See [Classifying Images with ImageNet](docs/imagenet-console.md)

#### Using the Console Program on Jetson

See [Classifying Images with ImageNet](docs/imagenet-console.md#using-the-console-program-on-jetson)

### Running the Live Camera Recognition Demo

See [Running the Live Camera Recognition Demo](docs/imagenet-camera.md)

### Re-training the Network with DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md)

#### Downloading Image Recognition Dataset

See [Re-Training the Recognition Network](docs/imagenet-training.md#downloading-image-recognition-dataset)

#### Customizing the Object Classes

See [Re-Training the Recognition Network](docs/imagenet-training.md#customizing-the-object-classes)

#### Importing Classification Dataset into DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#importing-classification-dataset-into-digits)

#### Creating Image Classification Model with DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#creating-image-classification-model-with-digits)

#### Testing Classification Model in DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#testing-classification-model-in-digits)

#### Downloading Model Snapshot to Jetson

See [Downloading Model Snapshots to Jetson](docs/imagenet-snapshot.md)

### Loading Custom Models on Jetson

See [Loading Custom Models on Jetson](docs/imagenet-custom.md)

### Locating Object Coordinates using DetectNet

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md)

#### Detection Data Formatting in DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#detection-data-formatting-in-digits)

#### Downloading the Detection Dataset

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#downloading-the-detection-dataset)

#### Importing the Detection Dataset into DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#importing-the-detection-dataset-into-digits)

#### Creating DetectNet Model with DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#creating-detectnet-model-with-digits)

#### Selecting DetectNet Batch Size

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#selecting-detectnet-batch-size)

#### Specifying the DetectNet Prototxt 

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#specifying-the-detectnet-prototxt)

#### Training the Model with Pretrained Googlenet

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#training-the-model-with-pretrained-googlenet)

#### Testing DetectNet Model Inference in DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#testing-detectnet-model-inference-in-digits)

#### Downloading the Model Snapshot to Jetson

See [Downloading the Detection Model to Jetson](docs/detectnet-snapshot.md)

#### DetectNet Patches for TensorRT

See [Downloading the Detection Model to Jetson](docs/detectnet-snapshot.md#detectnet-patches-for-tensorrt)

### Processing Images from the Command Line on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md)

#### Launching With a Pretrained Model

See [Detecting Objects from the Command Line](docs/detectnet-console.md#launching-with-a-pretrained-model)

#### Pretrained DetectNet Models Available

See [Detecting Objects from the Command Line](docs/detectnet-console.md#pretrained-detectnet-models-available)

#### Running Other MS-COCO Models on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md#running-other-ms-coco-models-on-jetson)

#### Running Pedestrian Models on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md#running-pedestrian-models-on-jetson)

#### Multi-class Object Detection Models

See [Detecting Objects from the Command Line](docs/detectnet-console.md#multi-class-object-detection-models)

### Running the Live Camera Detection Demo on Jetson

See [Running the Live Camera Detection Demo](docs/detectnet-camera.md)

### Image Segmentation with SegNet

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md)

#### Downloading Aerial Drone Dataset

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md#downloading-aerial-drone-dataset)

#### Importing the Aerial Dataset into DIGITS

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md#importing-the-aerial-dataset-into-digits)

#### Generating Pretrained FCN-Alexnet

See [Generating Pretrained FCN-Alexnet](docs/segnet-pretrained.md)

### Training FCN-Alexnet with DIGITS

See [Training FCN-Alexnet with DIGITS](docs/segnet-training.md)

#### Testing Inference Model in DIGITS

See [Training FCN-Alexnet with DIGITS](docs/segnet-training.md#testing-inference-model-in-digits)

#### FCN-Alexnet Patches for TensorRT

See [FCN-Alexnet Patches for TensorRT](docs/segnet-patches.md)

### Running Segmentation Models on Jetson

See [Running Segmentation Models on Jetson](docs/segnet-console.md)

</details>

##
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="#deploying-deep-learning"><sup>Table of Contents</sup></a></p>
