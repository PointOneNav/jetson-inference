

download_archive(){
    if [ ! -e $2 ];
    then
	echo "Downloading $2"
	wget --no-check-certificate $1 -O $2
	tar -xzvf $2
    else
	echo "Model already downloaded: " $2
    fi
    
}

download_archive 'https://nvidia.box.com/shared/static/mh121fvmveemujut7d8c9cbmglq18vz3.gz' "FCN-Alexnet-Cityscapes-HD.tar.gz"
download_archive "https://nvidia.box.com/shared/static/pa5d338t9ntca5chfbymnur53aykhall.gz" "FCN-Alexnet-Cityscapes-SD.tar.gz"
download_archive "https://nvidia.box.com/shared/static/xj20b6qopfwkkpqm12ffiuaekk6bs8op.gz" "FCN-Alexnet-Pascal-VOC.tar.gz"
download_archive "https://nvidia.box.com/shared/static/k7s7gdgi098309fndm2xbssj553vf71s.gz" "FCN-ResNet18-Cityscapes-512x256.tar.gz"
download_archive "https://nvidia.box.com/shared/static/9aqg4gpjmk7ipz4z0raa5mvs35om6emy.gz" "FCN-ResNet18-Cityscapes-1024x512.tar.gz"
download_archive "https://nvidia.box.com/shared/static/ylh3d2qk8qvitalq8sy803o7avrb6w0h.gz" "FCN-ResNet18-Cityscapes-2048x1024.tar.gz"
download_archive "https://nvidia.box.com/shared/static/jm0zlezvweiimpzluohg6453s0u0nvcv.gz" "FCN-ResNet18-DeepScene-576x320.tar.gz"
download_archive "https://nvidia.box.com/shared/static/gooux9b5nknk8wlk60ou9s2unpo760iq.gz" "FCN-ResNet18-DeepScene-864x480.tar.gz"
download_archive "https://nvidia.box.com/shared/static/dgaw0ave3bdws1t5ed333ftx5dbpt9zv.gz" "FCN-ResNet18-MHP-512x320.tar.gz"
download_archive "https://nvidia.box.com/shared/static/50mvlrjwbq9ugkmnnqp1sm99g2j21sfn.gz" "FCN-ResNet18-MHP-640x360.tar.gz"
download_archive "https://nvidia.box.com/shared/static/p63pgrr6tm33tn23913gq6qvaiarydaj.gz" "FCN-ResNet18-Pascal-VOC-320x320.tar.gz"
download_archive "https://nvidia.box.com/shared/static/njup7f3vu4mgju89kfre98olwljws5pk.gz" "FCN-ResNet18-Pascal-VOC-512x320.tar.gz" 
download_archive "https://nvidia.box.com/shared/static/5vs9t2wah5axav11k8o3l9skb7yy3xgd.gz" "FCN-ResNet18-SUN-RGBD-512x400.tar.gz"
download_archive "https://nvidia.box.com/shared/static/z5llxysbcqd8zzzsm7vjqeihs7ihdw20.gz" "FCN-ResNet18-SUN-RGBD-640x512.tar.gz" 
download_archive "https://nvidia.box.com/shared/static/u5ey2ws0nbtzyqyftkuqazx1honw6wry.gz" "FCN-Alexnet-SYNTHIA-CVPR16.tar.gz"
download_archive "https://nvidia.box.com/shared/static/vbk5ofu1x2hwp9luanbg4o0vrfub3a7j.gz" "FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz" 
download_archive "https://nvidia.box.com/shared/static/ydgmqgdhbvul6q9avoc9flxr3fdoa8pw.gz" "FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz"
download_archive "https://nvidia.box.com/shared/static/ht46fmnwvow0o0n0ke92x6bzkht8g5xb.gz" "ResNet-50.tar.gz"
