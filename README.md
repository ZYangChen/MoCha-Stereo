# MoCha-Stereo
[CVPR2024] The official implementation of "MoCha-Stereo: Motif Channel Attention Network for Stereo Matching".

<div align="center">
    <a href="https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers" target='_blank'><img src="https://img.shields.io/badge/CVPR-2024-9cf?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAjCAMAAAADt7LEAAADAFBMVEVMaXF5mcqXsNZ0lceQq9OXsNaHpM9zlMeHo8+dtNh9nMypvt12lsiFos6mu9xxksZvkcZzlMdlicJxksZqjcRukMUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACreRF2AAAAEXRSTlMA1mzoi0mi9BQ1wiRqVgeN+nWbeCoAAAAJcEhZcwAALiMAAC4jAXilP3YAAAJASURBVDiNnZTbduIwDEW3ZJOEwBTo/P8ntnRCCQHb8jzkUhgIrDV6iRN7W0eXCP7HZFotJKiPtsjh7pDfHq5f68YPq1/tNksEFfmsD/9iVt/e0lNvsaxivw/vuV01zxUqQHWsU5w+RbPnDHjg7bBLgCqmOZt854euguRhtfBAt8ugshfxMedNd3l4fyvdta/FWjKaDwkCVF/FK309haGyH4Lp6J4TAMqyFsjNywzcUsEMlcXLk8sbhYUmpOkz8BYA9DhtVwJk0wSklQGnnjotgaFaTV05U+w0UrZ2pG8DkLUqNFUHKGqQV8OpAMSUJlVvzkxdJX0wU/mVsRN7qsuGbMYYkkWTj5OLAGZmcaSigkyRBMBGZ6vagQl3ppQKNjkMrQPZ9IrPGvF/1uPWd4yxKPpsXCpwqRprGwDr/7EquEiSoSlbCdOfp4hC3Ewtf+Xssol4K+8FooQPB243lbmPLABZIbXHB5QDLcG0dEOrG2XGVxJ0Z6iLcah1kHhNZVmAOQdFLVV5oQDLrS2LDIczULl8Sykg5iCm1XZ5XhYtoXPgki4TqhngmLk1B4RUuAxmWkiuzrlSsLoAtD1DH8OdL8Lp4BQwMzRz2DswN176MIcAp7JR5xVQ2SlrAwzcx92M+1EInJMcj6U67LNw4dKt+kjO/UNLFXHxR+F1k1USfe4AdJsB/znMxdXFE/2ieUhdmfxOIF9zY0FnKAN/nJ1WM5TtHSnNTqsZCjF1j/r2hcn7k2k65wuZq/BTW/knm38BWrgDGcRH1DMAAAAASUVORK5CYII="/></a>&nbsp;
    <a href="https://arxiv.org/pdf/2404.06842.pdf" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp;
    <a href="" target='_blank'><img src="https://img.shields.io/badge/Supp.-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>&nbsp; <br>
    <a href="https://paperswithcode.com/sota/stereo-disparity-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-disparity-estimation-on-kitti-2015" /></a>
	<a href="https://paperswithcode.com/sota/stereo-depth-estimation-on-kitti-2015?p=mocha-stereo-motif-channel-attention-network" target='_blank'><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mocha-stereo-motif-channel-attention-network/stereo-depth-estimation-on-kitti-2015" /></a>
	
</div>

> MoCha-Stereo: Motif Channel Attention Network for Stereo Matching <br>
> [Ziyang Chen](https://orcid.org/0000-0002-9361-0240)†, [Wei Long](https://orcid.org/0000-0002-4121-2742)†, [He Yao](https://orcid.org/0009-0002-4212-5023)†, [Yongjun Zhang](http://cs.gzu.edu.cn/2021/1210/c17588a163831/page.htm)✱,[Bingshu Wang](https://teacher.nwpu.edu.cn/wangbingshu.html), [Yongbin Qin](http://cs.gzu.edu.cn/2021/1210/c17588a163794/page.htm), [Jia Wu](https://faculty.csu.edu.cn/jiawu/zh_CN/index.htm) <br>
> CVPR 2024 <br>
> Correspondence: ziyangchen2000@gmail.com; zyj6667@126.com✱ <br>
> Grateful to Prof. [Wenting Li](https://www.gzcc.edu.cn/jsjyxxgcxy/contents/3205/3569.html), Prof. [Huamin Qu](http://www.huamin.org/), and anonymous reviewers for their comments on this work.


https://github.com/ZYangChen/MoCha-Stereo/assets/108012397/2ed414fe-d182-499b-895c-b5375ef51425

```bibtex
@inproceedings{chen2024mocha,
  title={MoCha-Stereo: Motif Channel Attention Network for Stereo Matching},
	author={Chen, Ziyang and Long, Wei and Yao, He and Zhang, Yongjun and Wang, Bingshu and Qin, Yongbin and Wu, Jia},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	year={2024}
}
```
or
```bibtex
@article{chen2024mocha,
  title={MoCha-Stereo: Motif Channel Attention Network for Stereo Matching},
  author={Chen, Ziyang and Long, Wei and Yao, He and Zhang, Yongjun and Wang, Bingshu and Qin, Yongbin and Wu, Jia},
  journal={arXiv preprint arXiv:2404.06842},
  year={2024}
}
```
## Todo List
 - [CVPR2024] V1 version
    - [X] Preprint paper
    - [ ] Code of MoCha-Stereo (1. MoCha-Stereo will be made public in this repository in <strong>July, 2024</strong>. 2. For researchers at Guizhou University, I have made the code available in [our internal repository](https://github.com/GZU-ZhangYJ-group/mocha-stereo-early-access). Therefore, you do not need to contact me to get the code, just request access to the repository.)
    - [ ] Code of MoCha-MVS
          
<strong>The code and checkpoints are still being prepared. They will be released when they are sorted out!</strong>

## Acknowledgements
This project borrows the code from [<strong>IGEV</strong>](https://github.com/gangweiX/IGEV), [DLNR](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [GwcNet](https://github.com/xy-guo/GwcNet). We thank the original authors for their excellent works!
