# Installation

This guide will install:
* [CUDA-Toolkit](#cuda) release `11.2, V11.2.67`;
* [CUDNN](#cudnn) version `8.1.1`;
* [PyTorch](#torch) version `1.8.0`.

on an Ubuntu computer with the following specifications:
* the command `$ uname -m && cat /etc/*release` outputs:
  ```
  x86_64
  DISTRIB_ID=Ubuntu
  DISTRIB_RELEASE=20.04
  DISTRIB_CODENAME=focal
  DISTRIB_DESCRIPTION="Ubuntu 20.04.3 LTS"
  NAME="Ubuntu"
  VERSION="20.04.3 LTS (Focal Fossa)"
  ID=ubuntu
  ID_LIKE=debian
  PRETTY_NAME="Ubuntu 20.04.3 LTS"
  VERSION_ID="20.04"
  HOME_URL="https://www.ubuntu.com/"
  SUPPORT_URL="https://help.ubuntu.com/"
  BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
  PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
  VERSION_CODENAME=focal
  UBUNTU_CODENAME=focal
  ```
 * the command `$ gcc --version` outputs:
   ```
   gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
   Copyright (C) 2019 Free Software Foundation, Inc.
   This is free software; see the source for copying conditions.  There is NO
   warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
   ```
 * the command `$ uname -r` outputs:
   ```
   5.11.0-38-generic
   ```
 * the command `$ sudo lshw -C display` outputs:
   ```
   *-display                 
         description: VGA compatible controller
         product: GP104 [GeForce GTX 1080]
         vendor: NVIDIA Corporation
         physical id: 0
         bus info: pci@0000:09:00.0
         version: a1
         width: 64 bits
         clock: 33MHz
         capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
         configuration: driver=nvidia latency=0
         resources: irq:67 memory:fa000000-faffffff memory:c0000000-cfffffff memory:d0000000-d1ffffff ioport:e000(size=128) memory:fb000000-fb07ffff

   ```

## CUDA

Install **CUDA** by following [this guide](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#ubuntu-installation) and by downloading the version `11.2` from the [toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).  
It is **important** to follow the [post-installation](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#post-installation-actions) guide as well!  

Once the installation is completed, perform the [recommended actions](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-linux/index.html#recommended-post) to verify that all is done correctly. Note that the executable `deviceQuery` is found under `1_Utilities/deviceQuery`. Verify that the outcome is something like:
```
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.2, CUDA Runtime Version = 11.2, NumDevs = 3
Result = PASS
```

Check current installation with the command `$nvidia-smi` which shoudl output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:05:00.0  On |                  N/A |
| 28%   25C    P8    15W / 180W |    363MiB /  8116MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:06:00.0 Off |                  N/A |
| 28%   23C    P8     9W / 180W |     13MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 1080    Off  | 00000000:09:00.0 Off |                  N/A |
| 28%   23C    P8     9W / 180W |     13MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1057      G   /usr/lib/xorg/Xorg                 53MiB |
|    0   N/A  N/A      1700      G   /usr/lib/xorg/Xorg                157MiB |
|    0   N/A  N/A      1834      G   /usr/bin/gnome-shell               44MiB |
|    0   N/A  N/A     50998      G   ...AAAAAAAAA= --shared-files       93MiB |
|    1   N/A  N/A      1057      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A      1700      G   /usr/lib/xorg/Xorg                  4MiB |
|    2   N/A  N/A      1057      G   /usr/lib/xorg/Xorg                  4MiB |
|    2   N/A  N/A      1700      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```
Note that the `nvcc` version can be checked by `nvcc --version` which will report:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```


## CUDNN
This step requires to be registered as NVIDIA Developer Program Membership.  
Access the [CUDNN archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the following debian packages:
* `cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)`
* `cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)`
* `cuDNN Code Samples and User Guide for Ubuntu20.04 x86_64 (Deb)`  

Finally, follow the debian [installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb) also reported below for brevity:
* `$sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb`
* `$sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb`
* `$sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb`

Once the installation is complete it is possible to verify it by following this [guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify). The test should output something like this:
```
Result of classification: 1 3 5

Test passed!
```

The headers are located in `/usr/include` and it is possible to grep the version with `cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2` which should output:
```python
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 1
#define CUDNN_PATCHLEVEL 1
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H */
```

## Torch 

Install **PyTorch** by running (see [this guide](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1) for reference):  
`pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`.  
Older versions of `pytorch` are found [here](https://pytorch.org/get-started/previous-versions/).  

Verify the installation by running `python3` on the terminal, then executing:  
```python
import torch
torch.cuda.is_available()
print(torch.__version__)
print(torch.version.cuda)
```
 if the outputs are: `True`, `1.8.0+cu111`, `11.1` (to check if it still works with v11.2), then the installation is successful.

It is also required to install **Torch** to use the `th` command. The official [guide](http://torch.ch/docs/getting-started.html) does not seem to work with the CUDA versions installed above. However it is possible to install a working version supported by CUDA `10.0` and following.
```
git clone https://github.com/nagadomi/distro.git torch-cuda-10 --recursive
cd torch-cuda-10
./clean.sh
./update.sh
```

To check the installation:
```
$ ./torch-cuda-10/install/bin/th
 
  ______             __   |  Torch7 
 /_  __/__  ________/ /   |  Scientific computing for Lua. 
  / / / _ \/ __/ __/ _ \  |  Type ? for help 
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch 
                          |  http://torch.ch 
	
th>
```
Add the following lines to `~/.bashrc` to add the `th` command to the path and make it available from the `home` folder:
```bash
export PATH=$PATH:$HOME/torch-cuda-10/install/bin
```


## Troubleshooting

1. While training models or using pretrained ones, pytorch will download or access github libraries and put them under `/home/<usr>/.cache/torch/hub`. This can create some import errors as all repositories interfacing with pytorch will have the same structure. This is caused by the loading command `model, postprocessor = torch.hub.load('<github-repo:main>', '<model-name>', pretrained=True, return_postprocessor=True)`. This can be solved by reworking the repository saved in `/home/<usr>/.cache/torch/hub` by using specific imports. For example, the folders _configs_, _datasets_, _models_, _scripts_ and _utils_ in `/home/<usr>/.cache/torch/hub/ashkamath_mdetr_main` can be placed inside another folder (e.g. _ashk_) with an `__init__.py` file and the python scripts can be modified to import from that _ashk_ module instead. The same can be done for the Facebook repository in `/home/<usr>/.cache/torch/hub/facebookresearch_detr_master`.  

2. The library `huggingface-hub` will try to import `TypeAlias` from `typing_extensions` which will cause an error. This is solved by editing the file `repocard_types.py` located at `/home/<usr>/.local/lib/python3.8/site-packages/huggingface_hub`.  

3. Since Torch is meant to work with CUDA `10.0` some errors might show up while building the repository:
  * Error: `‘struct cudaPointerAttributes’ has no member named ‘isManaged’` is fixed by changing the function call `lua_pushboolean(L, attributes.type == cudaMemoryTypeManaged);` as in [source](https://githubmemory.com/repo/nagadomi/distro/issues/11).
  * Error: `identifier "cusparseScsrmm" is undefined` and `identifier "cusparseScsrmm"` are solved by commenting out the function calls with these two identifiers.
  * If `qt4` libraries are required, follow [this thread](https://askubuntu.com/questions/1234786/qt4-libqt4-in-ubuntu-20-04).
