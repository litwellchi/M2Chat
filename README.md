## $M^{2}Chat$: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation
The official release of $M^{2}Chat$. The published version code is still under development. NOT IMPLEMENTED YET
For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2311.17963).
Or [demo page](https://mattie-e.github.io/M2Chat.github.io/).

[![arXiv](https://img.shields.io/badge/arXiv-2311.17963-b31b1b.svg)](https://arxiv.org/abs/2311.17963)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://mattie-e.github.io/M2Chat.github.io/)

<img src="figs/main_banner.png" width="1000" >

<img src="figs/main_framework.png" width="1000" >

##TODO
- [ ] Release online demo
- [ ] Release checkpoint
- [ ] Update the M2chat adapter code
- [x] Upload mmdialog finetunning code
- [x] Upload training code
## Updates!!
* 【2024/04/15】 We update our experiment codes.
* 【2024/03/25】 We update our official papers on Arxiv.
* 【2023/11/29】 We publish our official papers on Arxiv.
## Quick Start
### Installation
**Step 0.** Install requirements.
```shell
pip install -r requirements.txt
```

**Step 1.** Download Llama2 pretrained weights ...
**Step 2.** Download Diffuser pretrained weight ...
**Step 3.** Download M2chat bias and querry ckpts  ...

### Notification
The publish version code is still under development. 
### Tutorials
**Validation.**
TODO

## Cite $M^{2}Chat$
If you use $M^{2}Chat$ in your research, please cite our work by using the following BibTeX entry:
```
@misc{chi2024m2chat,
      title={M$^{2}$Chat: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation}, 
      author={Xiaowei Chi and Rongyu Zhang and Zhengkai Jiang and Yijiang Liu and Yatian Wang and Xingqun Qi and Wenhan Luo and Peng Gao and Shanghang Zhang and Qifeng Liu and Yike Guo},
      year={2024},
      eprint={2311.17963},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{chi2023chatillusion,
  title={ChatIllusion: Efficient-Aligning Interleaved Generation ability with Visual Instruction Model},
  author={Chi, Xiaowei and Liu, Yijiang and Jiang, Zhengkai and Zhang, Rongyu and Lin, Ziyi and Zhang, Renrui and Gao, Peng and Fu, Chaoyou and Zhang, Shanghang and Liu, Qifeng and others},
  journal={arXiv preprint arXiv:2311.17963},
  year={2023}
}
```
## Thanks
We highly appreciate the effort of Llama-AdapterV2 and Stable Diffusion XL.

```latex
```
