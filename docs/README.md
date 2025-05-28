<div style="display: flex; align-items: center;">
  <img src="assets/images/prisma.jpg" alt="Vit Prisma Logo" style="margin-right: 10px;"/>
</div>

# Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video
This repo contains code for vision and video mechanistic interpretability, including activation caching and SAE training. We support a variety of vision/video models from Huggingface and OpenCLIP.

Mechanistic interpretability is currently split into two parts: circuit-analysis and sparse autoencoders (SAEs). Circuit-analysis finds the causal links between internal components of the model and primarily relies on activation caching. SAEs are like a more fine-grained "primitive" that you can use to examine intermediate activations. Prisma has the infrastructure to do both.

We also include a suite of [open source SAEs for all layers of CLIP and DINO](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/docs/sae_table.md) that you can download from Huggingface.

**For more details, check out our whitepaper [Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video](https://arxiv.org/abs/2504.19475).** Also, check out the original Less Wrong post [here](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).

**Table of Contents**
- [Installation](#Installation)
- [Vision SAE Demo Notebooks](#SAE-Demo-Notebooks)
- [Vision SAE Pretrained Weights)(#Pretrained-Vision-SAE-Suite)
- [Basic Mechanistic Interpretability](#Basic-Mechanistic-Interpretability)
- [Models Supported](#Models-Supported)
- [Contributors](#Contributors)
- [Citation](#Citation)

# Installation

We recommend installing from source:

For the latest version, install the repo from the source. While this version will include the latest developments, they may not be fully tested.

**Install from source**
To install as an editable repo from source:
```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

# SAE Pretrained Weights and Evaluation Code

#SAE Demo Notebooks
Here are notebooks to load, train, and evaluate SAEs.
* [SAE loading notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/1_Load_SAE.ipynb)
* [SAE training notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/2_Train_SAE.ipynb)
* [SAE evaluation notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/3_Evaluate_SAE.ipynb), which includes L0, cosine similarity, and reconstruction loss.

# Pretrained Vision SAE Suite
For a full list of SAEs for all layers, including CLIP top k, CLIP transcoders, and DINO SAEs, **see [here](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/docs/sae_table.md)**. More details are in our whitepaper [here](https://arxiv.org/abs/2504.19475).

We recommend starting with the vanilla CLIP SAEs, which are the highest quality. If you are just getting started with steering CLIP's output, we recommend using the [Layer 11 resid-post SAE](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05).

## [Experimental] Basic Mechanistic Interpretability
An earlier rendition of Prisma included features for basic mechanistic interpretability, including the logit lens and attention head visualizations. In addition to the tutorial notebooks below, you can also check out this [corresponding talk](https://youtu.be/gQbh-RZtsq4?t=0) on some of these techniques.

1. [Main ViT Demo](https://colab.research.google.com/drive/1TL_BY1huQ4-OTORKbiIg7XfTyUbmyToQ) - Overview of main mechanistic interpretability technique on a ViT, including direct logit attribution, attention head visualization, and activation patching. The activation patching switches the net's prediction from tabby cat to Border collie with a minimum ablation.
2. [Emoji Logit Lens](https://colab.research.google.com/drive/1yAHrEoIgkaVqdWC4GY-GQ46ZCnorkIVo) - Deeper dive into layer- and patch-level predictions with interactive plots.
3. [Interactive Attention Head Tour](https://colab.research.google.com/drive/1P252fCvTHNL_yhqJDeDVOXKCzIgIuAz2) - Deeper dive into the various types of attention heads a ViT contains with interactive JavaScript.

### Features
For a full demo of Prisma's features, including the visualizations below with interactivity, check out the demo notebooks above.

**Attention head visualization**

<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">

<div style="display: flex; align-items: center;">
  <img src="assets/images/attention head 1.png" alt="Logo Image 1" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 2.png" alt="Logo Image 2" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 3.png" alt="Logo Image 3" width="250"/>
</div>

**Activation patching**
<img src="assets/images/patched head.png" width="400">

**Direct logit attribution**
<img src="assets/images/direct logit attribution.png" width="600">

**Emoji logit lens**
<div style="display: flex; align-items: center;">
<img src="assets/images/dogit lens 2.png" width="400">
<img src="assets/images/cat toilet segmentation.png" width="400">
<img src="assets/images/child lion segmentation.png" width="400">
<img src="assets/images/cheetah segmentation.png" width="400">

</div>


## Models Supported
* [timm ViTs](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip)
* Our custom toy models (see below)


## Custom Models & Checkpoints

### ImageNet-1k classification checkpoints (patch size 32)

This model was trained by Praneet Suresh. All models include training checkpoints, in case you want to analyze training dynamics.

This larger patch size ViT has inspectable attention heads; else the patch size 16 attention heads are too large to easily render in JavaScript.


| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **3**         | 0.22 \| 0.42 |            N/A            | [Attention+MLP](https://huggingface.co/PraneetNeuro/ImageNet-Small-Attention-and-MLP-Patch32) |

### ImageNet-1k classification checkpoints (patch size 16)

The detailed training logs and metrics can be found [here](https://wandb.ai/vit-prisma/Imagenet/overview?workspace=user-yash-vadi). These models were trained by Yash Vadi.

**Table of Results**

Accuracy `[ <Acc> | <Top5 Acc> ]`

| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.16 \| 0.33             |  0.11 \| 0.25             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.23 \| 0.44             |  0.16 \| 0.34             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-base-Attention-and-MLP) |
| **small**| **3**         | 0.28 \| 0.51            | 0.17 \| 0.35             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-small-Attention-and-MLP) |
| **medium**|**4**         | 0.33 \| 0.56             | 0.17 \| 0.36             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-medium-Attention-and-MLP) |


# Contributors
Thank you to all our fantastic contributors! [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi), [Rob Graham](https://github.com/themachinefan), [Lorenz Hufe](https://github.com/lowlorenz), [Edward Stevinson](https://github.com/stevinson), and [Ethan Goldfarb](https://github.com/ekg15), _and more coming soon_. You learn more about our contributions on our [Contributors page](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/PRISMA_CONTRIBUTORS.md) (coming soon).

We welcome new contributors. Check out our contributing guidelines [here](CONTRIBUTING.md) and our [open Issues](https://github.com/soniajoseph/ViT-Prisma/issues).

# Citation

Please cite this repository when used in papers or research projects.
```
@misc{joseph2025prismaopensourcetoolkit,
      title={Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video}, 
      author={Sonia Joseph and Praneet Suresh and Lorenz Hufe and Edward Stevinson and Robert Graham and Yash Vadi and Danilo Bzdok and Sebastian Lapuschkin and Lee Sharkey and Blake Aaron Richards},
      year={2025},
      eprint={2504.19475},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.19475}, 
}
```
```
@misc{joseph2023vit,
  author = {Sonia Joseph},
  title = {ViT Prisma: A Mechanistic Interpretability Library for Vision Transformers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/soniajoseph/vit-prisma}}
}
```

# License

We have an MIT license [here](https://github.com/soniajoseph/ViT-Prisma/blob/main/LICENSE).
