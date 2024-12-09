# MoCha-Stereo 抹茶算法
[CVPR2024] The official implementation of "MoCha-Stereo: Motif Channel Attention Network for Stereo Matching".

[Arxiv] The extension version of MoCha-Stereo. "Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph"

https://github.com/ZYangChen/MoCha-Stereo/assets/108012397/2ed414fe-d182-499b-895c-b5375ef51425

## V1 Version

<div align="left">
    <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Chen_MoCha-Stereo_Motif_Channel_Attention_Network_for_Stereo_Matching_CVPR_2024_paper.html" target='_blank'><img src="https://img.shields.io/badge/CVPR-2024-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAjCAMAAAADt7LEAAADAFBMVEVMaXF5mcqXsNZ0lceQq9OXsNaHpM9zlMeHo8+dtNh9nMypvt12lsiFos6mu9xxksZvkcZzlMdlicJxksZqjcRukMUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACreRF2AAAAEXRSTlMA1mzoi0mi9BQ1wiRqVgeN+nWbeCoAAAAJcEhZcwAALiMAAC4jAXilP3YAAAJASURBVDiNnZTbduIwDEW3ZJOEwBTo/P8ntnRCCQHb8jzkUhgIrDV6iRN7W0eXCP7HZFotJKiPtsjh7pDfHq5f68YPq1/tNksEFfmsD/9iVt/e0lNvsaxivw/vuV01zxUqQHWsU5w+RbPnDHjg7bBLgCqmOZt854euguRhtfBAt8ugshfxMedNd3l4fyvdta/FWjKaDwkCVF/FK309haGyH4Lp6J4TAMqyFsjNywzcUsEMlcXLk8sbhYUmpOkz8BYA9DhtVwJk0wSklQGnnjotgaFaTV05U+w0UrZ2pG8DkLUqNFUHKGqQV8OpAMSUJlVvzkxdJX0wU/mVsRN7qsuGbMYYkkWTj5OLAGZmcaSigkyRBMBGZ6vagQl3ppQKNjkMrQPZ9IrPGvF/1uPWd4yxKPpsXCpwqRprGwDr/7EquEiSoSlbCdOfp4hC3Ewtf+Xssol4K+8FooQPB243lbmPLABZIbXHB5QDLcG0dEOrG2XGVxJ0Z6iLcah1kHhNZVmAOQdFLVV5oQDLrS2LDIczULl8Sykg5iCm1XZ5XhYtoXPgki4TqhngmLk1B4RUuAxmWkiuzrlSsLoAtD1DH8OdL8Lp4BQwMzRz2DswN176MIcAp7JR5xVQ2SlrAwzcx92M+1EInJMcj6U67LNw4dKt+kjO/UNLFXHxR+F1k1USfe4AdJsB/znMxdXFE/2ieUhdmfxOIF9zY0FnKAN/nJ1WM5TtHSnNTqsZCjF1j/r2hcn7k2k65wuZq/BTW/knm38BWrgDGcRH1DMAAAAASUVORK5CYII="/></a>&nbsp;
    <a href="https://arxiv.org/pdf/2404.06842.pdf" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
    <a href="https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_MoCha-Stereo_Motif_Channel_CVPR_2024_supplemental.pdf" target='_blank'><img src="https://img.shields.io/badge/Supp.-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp; 
    <a href="https://paperswithcode.com/sota/stereo-disparity-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-kitti-2015" /></a>
	<!--<a href="https://paperswithcode.com/sota/stereo-depth-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-depth-estimation-on-kitti-2015" /></a>-->
	
</div>

> MoCha-Stereo: Motif Channel Attention Network for Stereo Matching <br>
> [Ziyang Chen](https://scholar.google.com/citations?user=t64KgqAAAAAJ&hl=en&oi=sra)†, [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ&hl=en)†, [He Yao](https://scholar.google.com/citations?user=c0qjMAMAAAAJ&hl=en)†, [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yongbin Qin](http://cs.gzu.edu.cn/2021/1210/c17588a163794/page.htm), [Jia Wu](https://faculty.csu.edu.cn/jiawu/zh_CN/index.htm) <br>
> CVPR 2024 <br>
> Contact us: ziyangchen2000@gmail.com; zyj6667@126.com✱



```bibtex
@inproceedings{chen2024mocha,
  title={MoCha-Stereo: Motif Channel Attention Network for Stereo Matching},
  author={Chen, Ziyang and Long, Wei and Yao, He and Zhang, Yongjun and Wang, Bingshu and Qin, Yongbin and Wu, Jia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27768--27777},
  year={2024}
}
```

## V2 Version

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

## FAQ
**Q1: Weight for "tf_efficientnetv2_l"?** (Please refer to issue [#6 "关于tf_efficientnetv2_l检查点的问题"](https://github.com/ZYangChen/MoCha-Stereo/issues/6), [#8 "预训练权重"](https://github.com/ZYangChen/MoCha-Stereo/issues/8), and [#9 "code error"](https://github.com/ZYangChen/MoCha-Stereo/issues/9). )

A1: You can download it [here](https://github.com/ZYangChen/MoCha-Stereo/releases/download/tf_efficientnetv2_l-d664b728/tf_efficientnetv2_l-d664b728.pth). Moreover, the weight pretrained on the Scene Flow dataset is available [here](https://github.com/ZYangChen/MoCha-Stereo/releases/tag/MoCha-Stereo-V1).

**Q2: How to visualize the disparity map?** (Please refer to issue [#15 "询问可视化问题"](https://github.com/ZYangChen/MoCha-Stereo/issues/15), and [#10 "请问这个项目如何可视化推理的结果呢"](https://github.com/ZYangChen/MoCha-Stereo/issues/10))

A2: You can accomplish this using "[demo.py](https://github.com/ZYangChen/MoCha-Stereo/blob/main/MoCha-Stereo/demo.py)". 

```
python demo.py --restore_ckpt ./model/mocha-stereo.pth -l ./datasets/Middlebury/MiddEval3/trainingF/*/im0.png -r ./datasets/Middlebury/MiddEval3/trainingF/*/im1.png --output_directory ./your/path
```

The libary "matplotlib"  is required for visualizing the disparity map.

## Todo List
 - [CVPR2024] V1 version
    - [X] Paper
    - [X] Code of MoCha-Stereo
 - V2 version
    - [X] Preprint manuscript
    - [ ] Code of MoCha-V2


## Acknowledgements
<ul>
<li>This project borrows the code from <strong><a href="https://github.com/gangweiX/IGEV">IGEV</a></strong>, <a href="https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network">DLNR</a>, <a href="https://github.com/princeton-vl/RAFT-Stereo">RAFT-Stereo</a>, <a href="https://github.com/xy-guo/GwcNet">GwcNet</a>. We thank the original authors for their excellent works!</li>
<li>Grateful to Prof. <a href="https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html">Wenting Li</a>, Prof. <a href="http://www.huamin.org/">Huamin Qu</a>, Dr. <a href="https://github.com/Junda24">Junda Cheng</a>, Mr./Mrs. "DLUTTengYH", Mr./Mrs. "YHCks", and anonymous reviewers for their comments on "MoCha-Stereo: Motif Channel Attention Network for Stereo Matching" (V1 version of MoCha-Stereo).</li>
</ul>

