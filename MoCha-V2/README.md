# MoCha-V2 抹茶算法 Version 2

<a href="https://zyangchen.github.io/project_page/mocha_project/index.html"><img src="https://img.shields.io/badge/Project-Page-blue.svg"></a>
<a href="https://arxiv.org/abs/2411.12426" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-middlebury)](https://paperswithcode.com/sota/stereo-disparity-estimation-on-middlebury?p=mocha-stereo-motif-channel-attention-network)

> Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph <br>
> [Ziyang Chen](https://scholar.google.com/citations?user=t64KgqAAAAAJ&hl=en&oi=sra), [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Wenting Li](https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html), [Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yong Zhao](https://www.ece.pku.edu.cn/info/1045/2131.htm), [C. L. Philip Chen](https://www.ieeeiciea.org/2023/Prof.PhilipChen.html) <br>
> Arxiv Report <br>
> Contact us: ziyangchen2000@gmail.com; zyj6667@126.com✱

```bibtex
@article{chen2024motif,
  title={Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph},
  author={Chen, Ziyang and Zhang, Yongjun and Li, Wenting and Wang, Bingshu and Zhao, Yong and Chen, CL},
  journal={arXiv preprint arXiv:2411.12426},
  year={2024}
}
```

## V2 Version

### Dataset

To evaluate/train MoCha-stereo, you will need to download the required datasets. 
* [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) (Includes FlyingThings3D, Driving, Monkaa)
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [Falling Things (fat.zip)](https://research.nvidia.com/publication/2018-06_Falling-Things)
* [CARLA](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo/blob/master/dataset_download.sh)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)


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
        ├── 2005
        ├── 2006
        ├── 2014
        ├── 2021
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt
        ├── two_view_testing
    ├── TartanAir
    ├── fat
    ├── crestereo
    ├── HR-VS
        ├── carla-highres
    ├── InStereo2K

```

"official_train.txt" is available at [here](https://github.com/ZYangChen/MoCha-Stereo/issues/16).

