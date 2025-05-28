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
* [SAE loading notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/1_Load_SAE.ipynb) to load an SAE:
```
from huggingface_hub import hf_hub_download, list_repo_files
from vit_prisma.sae import SparseAutoencoder

repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05" # Change this to your chosen SAE. See /docs for a list of SAEs.

# Step 1: Download SAE from Hugginface
sae_path = hf_hub_download(repo_id, file_name='weights.pt') # Download weights
hf_hub_download(repo_id, config_name='config.pt') # Download config

# Step 2: Load the pretrained SAE weights from the downloaded path
print(f"Loading SAE from {sae_path}...")
sae = SparseAutoencoder.load_from_pretrained(sae_path) # This now automatically gets config.json and converts into the VisionSAERunnerConfig object

```
* [SAE training notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/2_Train_SAE.ipynb) to train an SAE:
```
# Step 1: Load model
from vit_prisma.models.model_loader import load_hooked_model

model_name = sae.cfg.model_name
model = load_hooked_model(model_name)
model.to(DEVICE)

# Step 2: Create Vision SAE Trainer Config
from vit_prisma.sae import VisionModelSAERunnerConfig
sae_trainer_cfg = VisionModelSAERunnerConfig( 
    hook_point_layer=10,
    layer_subtype='hook_resid_post',
    feature_sampling_window=1000,
    activation_fn_str='relu',
    checkpoint_path = '/network/scratch/s/sonia.joseph'
)

# Step 3: Load datsets
from vit_prisma.transforms import get_clip_val_transforms

imagenet_train_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train'
imagenet_validation_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val'

data_transforms = get_clip_val_transforms()
train_dataset = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)
eval_dataset = torchvision.datasets.ImageFolder(imagenet_validation_path, transform=data_transforms)

# Step 4: Train with Vision SAE Trainer Object
from vit_prisma.sae import VisionSAETrainer
trainer = VisionSAETrainer(sae_trainer_cfg, model, train_dataset, eval_dataset)
sae = trainer.run()
```
* [SAE evaluation notebook](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/demos/3_Evaluate_SAE.ipynb) to evaluate an SAE, including L0, cosine similarity, and reconstruction loss:
```
from vit_prisma.sae import SparsecoderEval
# Step 1: Load SAE and hooked vision/video model as in examples above

# Step 2: Run Sparsecoder Eval object
eval_runner = SparsecoderEval(sae, model) 
metrics = eval_runner.run_eval(is_clip=True)
```

# Pretrained Vision SAE Suite
For a full list of SAEs for all layers, including CLIP top k, CLIP transcoders, and DINO SAEs, **see [here](https://github.com/Prisma-Multimodal/ViT-Prisma/blob/main/docs/sae_table.md)**. More details are in our whitepaper [here](https://arxiv.org/abs/2504.19475).

We recommend starting with the vanilla CLIP SAEs, which are the highest fidelity. If you are just getting started with steering CLIP's output, we recommend using the [Layer 11 resid-post SAE](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05).

More statistics about the vanilla SAEs are below:

### Vanilla SAEs (All Patches)

| Model | Layer | Sublayer   | l1 coeff. | % Explained var. | Avg L0  | Avg CLS L0 | Cos sim | Recon cos sim | CE    | Recon CE | Zero abl CE | % CE recovered | % Alive features |
|--------|-------|------------|-----------|------------------|---------|-------------|---------|----------------|--------|-----------|--------------|----------------|------------------|
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_mlp_out-l1-1e-05) | 0     | mlp_out    | 1e-5      | 98.7             | 604.44  | 36.92       | 0.994   | 0.998          | 6.762  | 6.762     | 6.779        | 99.51          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_resid_post-l1-1e-05) | 0     | resid_post | 1e-5      | 98.6             | 1110.9  | 40.46       | 0.993   | 0.988          | 6.762  | 6.763     | 6.908        | 99.23          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_mlp_out-l1-1e-05) | 1     | mlp_out    | 1e-5      | 98.4             | 1476.8  | 97.82       | 0.992   | 0.994          | 6.762  | 6.762     | 6.889        | 99.40          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_resid_post-l1-1e-05) | 1     | resid_post | 1e-5      | 98.3             | 1508.4  | 27.39       | 0.991   | 0.989          | 6.762  | 6.763     | 6.908        | 99.02          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_mlp_out-l1-1e-05) | 2     | mlp_out    | 1e-5      | 98.0             | 1799.7  | 376.0       | 0.992   | 0.998          | 6.762  | 6.762     | 6.803        | 99.44          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_resid_post-l1-5e-05) | 2     | resid_post | 5e-5      | 90.6             | 717.84  | 10.11       | 0.944   | 0.960          | 6.762  | 6.767     | 6.908        | 96.34          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_mlp_out-l1-1e-05) | 3     | mlp_out    | 1e-5      | 98.1             | 1893.4  | 648.2       | 0.992   | 0.999          | 6.762  | 6.762     | 6.784        | 99.54          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_resid_post-l1-1e-05) | 3     | resid_post | 1e-5      | 98.1             | 2053.9  | 77.90       | 0.989   | 0.996          | 6.762  | 6.762     | 6.908        | 99.79          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_mlp_out-l1-1e-05) | 4     | mlp_out    | 1e-5      | 98.1             | 1901.2  | 1115.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.786        | 99.55          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_resid_post-l1-1e-05) | 4     | resid_post | 1e-5      | 98.0             | 2068.3  | 156.7       | 0.989   | 0.997          | 6.762  | 6.762     | 6.908        | 99.74          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_mlp_out-l1-1e-05) | 5     | mlp_out    | 1e-5      | 98.2             | 1761.5  | 1259.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.797        | 99.76          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_resid_post-l1-1e-05) | 5     | resid_post | 1e-5      | 98.1             | 1953.8  | 228.5       | 0.990   | 0.997          | 6.762  | 6.762     | 6.908        | 99.80          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_mlp_out-l1-1e-05) | 6     | mlp_out    | 1e-5      | 98.3             | 1598.0  | 1337.0      | 0.993   | 0.999          | 6.762  | 6.762     | 6.789        | 99.83          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_resid_post-l1-1e-05) | 6     | resid_post | 1e-5      | 98.2             | 1717.5  | 321.3       | 0.991   | 0.996          | 6.762  | 6.762     | 6.908        | 99.93          | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_mlp_out-l1-1e-05) | 7     | mlp_out    | 1e-5      | 98.2             | 1535.3  | 1300.0      | 0.992   | 0.999          | 6.762  | 6.762     | 6.796        | 100.17         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_resid_post-l1-1e-05) | 7     | resid_post | 1e-5      | 98.2             | 1688.4  | 494.3       | 0.991   | 0.995          | 6.762  | 6.761     | 6.908        | 100.24         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-1e-05) | 8     | mlp_out    | 1e-5      | 97.8             | 1074.5  | 1167.0      | 0.990   | 0.998          | 6.762  | 6.761     | 6.793        | 100.57         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-1e-05) | 8     | resid_post | 1e-5      | 98.2             | 1570.8  | 791.3       | 0.991   | 0.992          | 6.762  | 6.761     | 6.908        | 100.41         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_mlp_out-l1-1e-05) | 9     | mlp_out    | 1e-5      | 97.6             | 856.68  | 1076.0      | 0.989   | 0.998          | 6.762  | 6.762     | 6.792        | 100.28         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-1e-05) | 9     | resid_post | 1e-5      | 98.2             | 1533.5  | 1053.0      | 0.991   | 0.989          | 6.762  | 6.761     | 6.908        | 100.32         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_mlp_out-l1-1e-05) | 10    | mlp_out    | 1e-5      | 98.1             | 788.49  | 965.5       | 0.991   | 0.998          | 6.762  | 6.762     | 6.772        | 101.50         | 99.80            |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_resid_post-l1-1e-05) | 10    | resid_post | 1e-5      | 98.4             | 1292.6  | 1010.0      | 0.992   | 0.987          | 6.762  | 6.760     | 6.908        | 100.83         | 99.99            |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_mlp_out-l1-5e-05) | 11    | mlp_out    | 5e-5      | 89.7             | 748.14  | 745.5       | 0.972   | 0.993          | 6.762  | 6.759     | 6.768        | 135.77         | 100              |
| [link](https://huggingface.co/prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05) | 11    | resid_post | 1e-5      | 98.4             | 1405.0  | 1189.0      | 0.993   | 0.987          | 6.762  | 6.765     | 6.908        | 98.03          | 99.99            |



## Basic Mechanistic Interpretability
Check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).

Check out our tutorial notebooks for using the repo. You can also check out this [corresponding talk](https://youtu.be/gQbh-RZtsq4?t=0) on some of these techniques.

1. [Main ViT Demo](https://colab.research.google.com/drive/1TL_BY1huQ4-OTORKbiIg7XfTyUbmyToQ) - Overview of main mechanistic interpretability technique on a ViT, including direct logit attribution, attention head visualization, and activation patching. The activation patching switches the net's prediction from tabby cat to Border collie with a minimum ablation.
2. [Emoji Logit Lens](https://colab.research.google.com/drive/1yAHrEoIgkaVqdWC4GY-GQ46ZCnorkIVo) - Deeper dive into layer- and patch-level predictions with interactive plots.
3. [Interactive Attention Head Tour](https://colab.research.google.com/drive/1P252fCvTHNL_yhqJDeDVOXKCzIgIuAz2) - Deeper dive into the various types of attention heads a ViT contains with interactive JavaScript.

## Features

For a full demo of Prisma's features, including the visualizations below with interactivity, check out the demo notebooks above.

### Attention head visualization
<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">

<div style="display: flex; align-items: center;">
  <img src="assets/images/attention head 1.png" alt="Logo Image 1" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 2.png" alt="Logo Image 2" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 3.png" alt="Logo Image 3" width="250"/>
</div>

### Activation patching
<img src="assets/images/patched head.png" width="400">

### Direct logit attribution
<img src="assets/images/direct logit attribution.png" width="600">

### Emoji logit lens
<div style="display: flex; align-items: center;">
<img src="assets/images/dogit lens 2.png" width="400">
<img src="assets/images/cat toilet segmentation.png" width="400">
<img src="assets/images/child lion segmentation.png" width="400">
<img src="assets/images/cheetah segmentation.png" width="400">

</div>


## Supported Models
* [timm ViTs](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip)
* Our custom toy models (see below)

## Training Code

Prisma contains training code to train your own custom ViTs. Training small ViTs can be very useful when isolating specific behaviors in the model.

For training your own models, check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).

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

### dSprites Shape Classification training checkpoints

Original dataset is [here](https://github.com/google-deepmind/dsprites-dataset). 

Full results and training setup are [here](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/experiments/dSprites_results.md). These models were trained by Yash Vadi.

**Table of Results**
| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.535             | 0.459             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.996             | 0.685             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-base-Attention-and-MLP) |
| **small**| **3**         | 1.000             | 0.774             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-small-Attention-and-MLP) |
| **medium**|**4**         | 1.000             | 0.991             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-medium-Attention-and-MLP) |


## Guidelines for training + uploading models

Upload your trained models to Huggingface. Follow the [Huggingface guidelines](https://huggingface.co/docs/hub/models-uploading) and also create a model card. Document as much of the training process as possible including links to loss and accuracy curves on weights and biases, dataset (and order of training data), hyperparameters, optimizer, learning rate schedule, hardware, and other details that may be relevant. 

Include frequent checkpoints throughout training, which will help other researchers understand training dynamics.

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
