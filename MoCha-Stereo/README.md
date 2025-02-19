# MoCha-Stereo 抹茶算法

## V1 Version

> MoCha-Stereo: Motif Channel Attention Network for Stereo Matching <br>
> [Ziyang Chen](https://scholar.google.com/citations?user=t64KgqAAAAAJ&hl=en&oi=sra)†, [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ&hl=en)†, [He Yao](https://scholar.google.com/citations?user=c0qjMAMAAAAJ&hl=en)†, [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yongbin Qin](http://cs.gzu.edu.cn/2021/1210/c17588a163794/page.htm), [Jia Wu](https://faculty.csu.edu.cn/jiawu/zh_CN/index.htm) <br>
> CVPR 2024 <br>
> Correspondence: ziyangchen2000@gmail.com; zyj6667@126.com✱


<div align="center">
    <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Chen_MoCha-Stereo_Motif_Channel_Attention_Network_for_Stereo_Matching_CVPR_2024_paper.html" target='_blank'><img src="https://img.shields.io/badge/CVPR-2024-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAjCAMAAAADt7LEAAADAFBMVEVMaXF5mcqXsNZ0lceQq9OXsNaHpM9zlMeHo8+dtNh9nMypvt12lsiFos6mu9xxksZvkcZzlMdlicJxksZqjcRukMUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACreRF2AAAAEXRSTlMA1mzoi0mi9BQ1wiRqVgeN+nWbeCoAAAAJcEhZcwAALiMAAC4jAXilP3YAAAJASURBVDiNnZTbduIwDEW3ZJOEwBTo/P8ntnRCCQHb8jzkUhgIrDV6iRN7W0eXCP7HZFotJKiPtsjh7pDfHq5f68YPq1/tNksEFfmsD/9iVt/e0lNvsaxivw/vuV01zxUqQHWsU5w+RbPnDHjg7bBLgCqmOZt854euguRhtfBAt8ugshfxMedNd3l4fyvdta/FWjKaDwkCVF/FK309haGyH4Lp6J4TAMqyFsjNywzcUsEMlcXLk8sbhYUmpOkz8BYA9DhtVwJk0wSklQGnnjotgaFaTV05U+w0UrZ2pG8DkLUqNFUHKGqQV8OpAMSUJlVvzkxdJX0wU/mVsRN7qsuGbMYYkkWTj5OLAGZmcaSigkyRBMBGZ6vagQl3ppQKNjkMrQPZ9IrPGvF/1uPWd4yxKPpsXCpwqRprGwDr/7EquEiSoSlbCdOfp4hC3Ewtf+Xssol4K+8FooQPB243lbmPLABZIbXHB5QDLcG0dEOrG2XGVxJ0Z6iLcah1kHhNZVmAOQdFLVV5oQDLrS2LDIczULl8Sykg5iCm1XZ5XhYtoXPgki4TqhngmLk1B4RUuAxmWkiuzrlSsLoAtD1DH8OdL8Lp4BQwMzRz2DswN176MIcAp7JR5xVQ2SlrAwzcx92M+1EInJMcj6U67LNw4dKt+kjO/UNLFXHxR+F1k1USfe4AdJsB/znMxdXFE/2ieUhdmfxOIF9zY0FnKAN/nJ1WM5TtHSnNTqsZCjF1j/r2hcn7k2k65wuZq/BTW/knm38BWrgDGcRH1DMAAAAASUVORK5CYII="/></a>&nbsp;
    <a href="https://arxiv.org/pdf/2404.06842.pdf" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
    <a href="https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_MoCha-Stereo_Motif_Channel_CVPR_2024_supplemental.pdf" target='_blank'><img src="https://img.shields.io/badge/Supp.-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp; 
    <a href="https://paperswithcode.com/sota/stereo-disparity-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-kitti-2015" /></a>
	<!--<a href="https://paperswithcode.com/sota/stereo-depth-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-depth-estimation-on-kitti-2015" /></a>-->
	
</div>


```bibtex
@inproceedings{chen2024mocha,
  title={MoCha-Stereo: Motif Channel Attention Network for Stereo Matching},
  author={Chen, Ziyang and Long, Wei and Yao, He and Zhang, Yongjun and Wang, Bingshu and Qin, Yongbin and Wu, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27768--27777},
  year={2024}
}
```

### Requirements

Python = 3.8

CUDA = 11.3

```Shell
conda create -n mocha python=3.8
conda activate mocha
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

The following libraries are also required
```Shell
tqdm
tensorboard
opt_einsum
einops
scipy
imageio
opencv-python-headless
scikit-image
timm == 0.6.5
six
```

You can install them via
```Shell
pip install -r requirements.txt
```

### Dataset

To evaluate/train MoCha-stereo, you will need to download the required datasets. 
* [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) (Includes FlyingThings3D, Driving, Monkaa)
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)


By default `stereo_datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── FlyingThings3D
        ├── frames_finalpass
        ├── disparity
    ├── Monkaa
        ├── frames_finalpass
        ├── disparity
    ├── Driving
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── KITTI_2015
            ├── testing
            ├── training
        ├── KITTI_2012
            ├── testing
            ├── training
    ├── Middlebury
        ├── MiddEval3
		├── trainingF
		├── trainingH
		├── trainingQ
		├── official_train.txt
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt
        ├── two_view_testing

```

"official_train.txt" is available at [here](https://github.com/ZYangChen/MoCha-Stereo/issues/16).

### Training

```Shell
bash ./scripts/train.sh
```

### Evaluation

To evaluate a trained model on a validation set (e.g. Middlebury full resolution), run
```Shell
bash evaluate.sh
```

or

```Shell
python evaluate_stereo.py --restore_ckpt models/mocha-stereo.pth --dataset middlebury_F
```

Weight is available [here](https://github.com/ZYangChen/MoCha-Stereo/releases/download/MoCha-Stereo-V1/mocha-stereo.pth).

## Acknowledgements
<ul>
<li>This project is supported by Science and Technology Planning Project of Guizhou Province, Department of Science and Technology of Guizhou Province, China (Project No. [2023]159). </li>
<li>This project is supported by Natural Science Research Project of Guizhou Provincial Department of Education, China (QianJiaoJi[2022]029, QianJiaoHeKY[2021]022).</li>
<li>Grateful to Prof. <a href="https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html">Wenting Li</a>, Prof. <a href="http://www.huamin.org/">Huamin Qu</a>, Dr. <a href="https://github.com/Junda24">Junda Cheng</a>, Mr./Mrs. "DLUTTengYH", Mr./Mrs. "YHCks", and anonymous reviewers for their comments on "MoCha-Stereo: Motif Channel Attention Network for Stereo Matching" (V1 version of MoCha-Stereo).</li>
</ul>

