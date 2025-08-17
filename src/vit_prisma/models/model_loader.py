"""
Efficient Prisma Model Loading System
===================================

Configs and weights load directly from Huggingface, with config and weight conversion scripts
to make them compatible with the Prisma library.

"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch

import torch.nn as nn
from vit_prisma.configs.HookedTextTransformerConfig import HookedTextTransformerConfig

from vit_prisma.configs.HookedViTConfig import HookedViTConfig

# Import configuration dictionaries - these are just static data
from vit_prisma.models.model_config_registry import (
    MODEL_CATEGORIES,
    MODEL_CONFIGS,
    ModelCategory,
    TEXT_SUPPORTED_MODELS,
)

# Import conversion functions for reference only - will be loaded on demand
from vit_prisma.models.weight_conversion import (
    convert_clip_weights,
    convert_dino_weights,
    convert_kandinsky_clip_weights,
    convert_open_clip_text_weights,
    convert_open_clip_weights,
    convert_timm_weights,
    convert_vivet_weights,
    convert_vjepa_weights,
    download_pretrained_from_hf,
    load_state_dict,
    remove_open_clip_prefix,
)
from vit_prisma.utils.enums import ModelType

# Type alias
ConfigType = Union[HookedViTConfig, HookedTextTransformerConfig]


DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


MODELS_MISSING_CONFIG = {
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k": (
        "xlm-roberta-base-ViT-B-32",
        "laion5b_s13b_b90k",
    ),
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k": (
        "roberta-ViT-B-32",
        "laion2b_s12b_b32k",
    ),
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k": (
        "xlm-roberta-large-ViT-H-14",
        "frozen_laion5b_s13b_b90k",
    ),
    "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k": (
        "coca_ViT-B-32",
        "laion2b_s13b_b90k",
    ),
    "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k": (
        "coca_ViT-L-14",
        "laion2b_s13b_b90k",
    ),
}

# Models that pass and fail according to 'tests/test_loading_clip.py'. Please update list as models get fixed / break.
PASSING_MODELS = {
    "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
    "open-clip:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K",
    "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch16_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.laion2b_e16",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31",
    "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32",
    "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "facebook/dino-vitb16",
    "facebook/dino-vitb8",
    "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
}

FAILING_MODELS = {
    # # MODELS THAT FAIL CURRENTLY
    "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch16_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch16_clip_224.openai",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e31",
    "open-clip:timm/vit_base_patch32_clip_224.laion400m_e32",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_base_patch32_clip_224.metaclip_400m",
    "open-clip:timm/vit_base_patch32_clip_224.openai",
    "open-clip:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K",
    "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
    "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e31",
    "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e32",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_large_patch14_clip_224.metaclip_400m",
    "open-clip:timm/vit_large_patch14_clip_224.openai",
    "open-clip:timm/vit_large_patch14_clip_336.openai",
    "open-clip:timm/vit_medium_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m",
    "open-clip:timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m",
    "open-clip:timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b",
    "open-clip:timm/vit_huge_patch14_clip_224.metaclip_2pt5b",
    "facebook/dino-vits16",
    "facebook/dino-vits8",
}


# ===============================
# Core Model Loading Functions
# ===============================


def load_config(
    model_name: str,
    model_type: ModelType = ModelType.VISION,
    local_path: Optional[str] = None,
    **kwargs,
) -> ConfigType:
    """
    Load and create configuration for a model.

    Args:
        model_name: Name of the model to load
        model_type: Type of model (VISION or TEXT)
        local_path: Path to local model directory (for OpenCLIP models)
        **kwargs: Additional arguments

    Returns:
        Model configuration
    """
    if model_name not in MODEL_CATEGORIES:
        raise ValueError(f"Model '{model_name}' is not registered in configurations")

    if model_type == ModelType.TEXT and model_name not in TEXT_SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' does not support text modality")

    category = MODEL_CATEGORIES[model_name]

    # Get dynamic config from source
    if category == ModelCategory.TIMM:
        old_config = _get_timm_hf_config(model_name)
        new_config = _create_config_from_hf(old_config, model_name, model_type)
    elif category == ModelCategory.OPEN_CLIP:
        old_config = _get_open_clip_config(
            model_name, model_type, local_path=local_path
        )
        new_config = _create_config_from_open_clip(old_config, model_name, model_type)
    elif category == ModelCategory.DINO:
        old_config = _get_general_hf_config(model_name, model_type=None)
        new_config = _create_config_from_hf(old_config, model_name, model_type=None)
    elif category in [ModelCategory.CLIP, ModelCategory.VIVIT]:
        old_config = _get_general_hf_config(model_name, model_type)
        new_config = _create_config_from_hf(old_config, model_name, model_type)

    # Apply registry overrides
    registry_overrides = MODEL_CONFIGS[model_type].get(model_name, {})
    for key, value in registry_overrides.items():
        setattr(new_config, key, value)

    new_config.d_head = (
        new_config.d_model // new_config.n_heads
    )  # Calculate this after retrieving latest info
    return new_config


def check_model_name(model_name: str, allow_failing: bool = False) -> str:
    """
    Check if a model name is valid and supported.

    Args:
        model_name: Name of the model to check
        allow_failing: Whether to allow loading models that are known to fail tests

    Returns:
        Potentially modified model name
    """
    if model_name in MODELS_MISSING_CONFIG:
        model_name = MODELS_MISSING_CONFIG[model_name][0]
        logging.warning(
            f"Model '{model_name}' is missing a configuration in the registry. Using '{model_name}' instead."
        )

    if model_name in FAILING_MODELS:
        msg = f"Model '{model_name}' is in the list of models failing tests."
        if not allow_failing:
            raise ValueError(msg + " Set allow_failing=True to load anyway.")
        else:
            logging.warning(msg + " Loading anyway as allow_failing=True.")
    elif model_name in PASSING_MODELS:
        logging.info(f"Model '{model_name}' is supported and passes tests.")
    else:
        logging.warning(
            f"Model '{model_name}' is not in the lists of models passing or failing tests. Unclear status. You may want to check that the HookedViT matches the original model under tests/test_loading_clip.py."
        )

    return model_name


def load_weights(
    model: nn.Module,
    model_name: str,
    model_type: ModelType,
    dtype: torch.dtype,
    **kwargs,
) -> None:
    """
    Load and apply weights to a model.

    Args:
        model: Model to load weights into
        model_name: Name of the model
        model_type: Type of model (VISION or TEXT)
        dtype: Data type for weights
        **kwargs: Additional arguments
    """
    category = MODEL_CATEGORIES[model_name]
    config = model.cfg

    # Load and convert weights
    original_weights = load_original_weights(
        model_name, category, model_type, dtype, **kwargs
    )
    converted_weights = convert_weights(
        original_weights, model_name, category, config, model_type
    )

    # Apply weights to model
    full_state_dict = fill_missing_keys(model, converted_weights)

    return full_state_dict


def load_hooked_model(
    model_name: str,
    model_class: Type = None,
    model_type: ModelType = ModelType.VISION,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    pretrained: bool = True,
    fold_ln: bool = False,
    center_writing_weights: bool = False,
    fold_value_biases: bool = True,
    refactor_factored_attn_matrices: bool = False,
    move_to_device: bool = True,
    use_attn_result: bool = False,
    allow_failing: bool = False,
    local_path: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Load a model with configuration and weights.

    Args:
        model_name: Name of the model to load
        model_class: Class to instantiate (optional, will be inferred if None)
        model_type: Type of model (VISION or TEXT)
        device: Device to load model on
        dtype: Data type for model parameters
        pretrained: Whether to load pretrained weights
        fold_ln: Whether to fold layer normalization into attention
        center_writing_weights: Whether to center writing weights
        allow_failing: Whether to allow loading models that are known to fail tests
        local_path: Path to local model weights (for OpenCLIP models)
        **kwargs: Additional arguments

    Returns:
        Loaded model
    """
    assert not (
        kwargs.get("load_in_8bit", False) or kwargs.get("load_in_4bit", False)
    ), "Quantization not supported"

    if isinstance(dtype, str):
        # Convert from string to a torch dtype
        dtype = DTYPE_FROM_STRING[dtype]

    if "torch_dtype" in kwargs:
        # For backwards compatibility with the previous way to do low precision loading
        # This should maybe check the user did not explicitly set dtype *and* torch_dtype
        dtype = kwargs["torch_dtype"]

    if (
        (kwargs.get("torch_dtype", None) == torch.float16) or dtype == torch.float16
    ) and device in ["cpu", None]:
        logging.warning(
            "float16 models may not work on CPU. Consider using a GPU or bfloat16."
        )

    model_name = check_model_name(model_name, allow_failing)

    config = load_config(model_name, model_type, local_path, **kwargs)

    if model_class is None:
        if model_type == ModelType.VISION:
            from vit_prisma.models.base_vit import HookedViT

            model_class = HookedViT
        else:  # TEXT
            from vit_prisma.models.base_text_transformer import HookedTextTransformer

            model_class = HookedTextTransformer

    # Initialize model
    model = model_class(config)

    # Load weights if requested
    if pretrained:
        # Make sure local_path is included in kwargs for weight loading
        if local_path is not None:
            kwargs["local_path"] = local_path
        state_dict = load_weights(model, model_name, model_type, dtype, **kwargs)

    model.load_and_process_state_dict(
        state_dict,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
    )

    model.to(device=device, dtype=dtype)
    model.set_use_attn_result(use_attn_result)

    if move_to_device:
        model.move_model_modules_to_device()

    logging.info(f"Loaded pretrained model {model_name} into HookedTransformer")
    return model


def _get_general_hf_config(model_name: str, model_type=None):
    """Get HuggingFace config from TIMM model."""
    from transformers import AutoConfig

    if model_type:
        if model_type == ModelType.VISION:
            model_type = "vision_config"
        elif model_type == ModelType.TEXT:
            model_type = "text_config"
    hf_config = AutoConfig.from_pretrained(model_name)
    if model_type:
        hf_config = getattr(hf_config, model_type)
    return hf_config


def _get_timm_hf_config(model_name: str):
    """Get HuggingFace config from TIMM model."""
    import timm

    model = timm.create_model(model_name)
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model.default_cfg["hf_hub_id"])
    return hf_config


def _get_open_clip_config(model_name: str, model_type: ModelType, local_path=None):
    import json
    import os

    import open_clip

    if local_path is not None:
        # Look for config file in the local directory
        if os.path.isdir(local_path):
            config_path = os.path.join(local_path, "open_clip_config.json")
        else:
            # Assume local_path is the directory containing the model files
            config_path = os.path.join(
                os.path.dirname(local_path), "open_clip_config.json"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Please ensure 'open_clip_config.json' is in the same directory as your model weights."
            )
    else:
        # Download from HuggingFace
        config_path = download_pretrained_from_hf(
            remove_open_clip_prefix(model_name), filename="open_clip_config.json"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        model_config = config["model_cfg"]
    return model_config


def _create_config_from_open_clip(model_cfg, model_name, model_type: ModelType):

    cfg = HookedViTConfig()
    cfg.d_model = model_cfg["vision_cfg"]["width"]
    cfg.n_layers = model_cfg["vision_cfg"]["layers"]
    cfg.patch_size = model_cfg["vision_cfg"]["patch_size"]
    cfg.image_size = model_cfg["vision_cfg"]["image_size"]
    cfg.n_classes = model_cfg["embed_dim"]
    # cfg.n_heads = model_cfg['vision_cfg']['num_attention_heads']

    cfg.model_name = model_name

    # Attention head number is not in open clip config, so we add it manually
    if "plus_clip" in model_name:
        cfg.n_heads = 14
    elif any(s in model_name for s in ["vit_xsmall"]):
        cfg.n_heads = 8
    elif any(s in model_name for s in ["ViT-B", "vit-base"]):
        cfg.n_heads = 12
    elif any(s in model_name for s in ["ViT-L", "vit_large", "vit_medium", "bigG"]):
        cfg.n_heads = 16
    elif any(s in model_name for s in ["huge_", "ViT-H"]):
        cfg.n_heads = 20
    elif any(s in model_name for s in ["ViT-g", "giant_"]):
        cfg.n_heads = 22
    elif any(s in model_name for s in ["gigantic_"]):
        cfg.n_heads = 26
    else:
        cfg.n_heads = 12

    # Set MLP dimension
    if model_cfg["vision_cfg"].get("mlp_ratio"):
        cfg.d_mlp = int(cfg.d_model * model_cfg["vision_cfg"].get("mlp_ratio"))
    else:
        cfg.d_mlp = cfg.d_model * 4

    # Common configurations
    cfg.normalization_type = "LN"
    cfg.return_type = "class_logits"

    return cfg


def _create_config_from_hf(hf_config, model_name: str, model_type: ModelType):
    """Create a general config from HuggingFace config for any vision or text transformer."""
    if model_type == ModelType.VISION or model_type == None:  # VISION
        config = HookedViTConfig()

        # Core architecture parameters
        config.d_model = hf_config.hidden_size
        config.n_layers = hf_config.num_hidden_layers
        config.n_heads = hf_config.num_attention_heads
        config.d_head = hf_config.hidden_size // hf_config.num_attention_heads
        config.d_mlp = hf_config.intermediate_size

        # Vision-specific parameters
        config.image_size = getattr(hf_config, "image_size", 224)
        config.n_channels = getattr(hf_config, "num_channels", 3)
        config.patch_size = getattr(hf_config, "patch_size", 16)

        # Handle different types of patch sizes
        if hasattr(hf_config, "tubelet_size"):
            config.patch_size = hf_config.tubelet_size[1]
            config.is_video_transformer = True
            config.video_tubelet_depth = hf_config.tubelet_size[0]
            config.video_num_frames = hf_config.video_size[0]

    elif model_type == ModelType.TEXT:  # TEXT
        config = HookedTextTransformerConfig()
        config.d_model = hf_config.hidden_size
        config.n_layers = hf_config.num_hidden_layers
        config.n_heads = hf_config.num_attention_heads
        config.d_head = hf_config.hidden_size // hf_config.num_attention_heads
        config.d_mlp = hf_config.intermediate_size
        config.vocab_size = hf_config.vocab_size
        config.context_length = getattr(hf_config, "max_position_embeddings", 77)

    # Common parameters
    config.model_name = model_name
    config.initializer_range = getattr(hf_config, "initializer_range", 0.02)
    config.eps = getattr(hf_config, "layer_norm_eps", 1e-5)

    # Handle output dimension
    if hasattr(hf_config, "projection_dim"):
        config.n_classes = hf_config.projection_dim
        config.return_type = "class_logits"
    elif hasattr(hf_config, "num_classes"):
        config.n_classes = hf_config.num_classes
        config.return_type = "class_logits"
    else:
        config.n_classes = config.d_model
        config.return_type = "pre_logits"

    return config


# def _create_clip_config_from_hf(hf_config, model_name: str, model_type: ModelType):
#     """Create Prisma config from HuggingFace config."""
#     if model_type == ModelType.VISION:
#         config = HookedViTConfig()

#         # Extract patch size
#         if hasattr(hf_config, "patch_size"):
#             config.patch_size = hf_config.patch_size
#         elif hasattr(hf_config, "tubelet_size"):
#             config.patch_size = hf_config.tubelet_size[1]

#         # Common attributes
#         config.d_model = hf_config.vision_config.hidden_size
#         config.n_layers = hf_config.vision_config.num_hidden_layers
#         config.n_heads = hf_config.vision_config.num_attention_heads
#         config.d_head = hf_config.vision_config.hidden_size // hf_config.vision_config.num_attention_heads
#         config.d_mlp = hf_config.vision_config.intermediate_size
#         config.image_size = getattr(hf_config, "image_size", 224)
#         config.n_channels = getattr(hf_config, "num_channels", 3)
#         config.patch_size = getattr(hf_config, "patch_size", 16)
#         config.initializer_range = getattr(hf_config, "initializer_range", 0.02)

#         config.model_name = model_name

#         if hasattr(hf_config, "layer_norm_eps"):
#             config.eps = hf_config.layer_norm_eps


#         # Set output dimension appropriately
#         if hasattr(hf_config, "projection_dim"):
#             config.n_classes = hf_config.projection_dim
#             config.return_type = "class_logits"
#         elif hasattr(hf_config, "num_classes"):
#             config.n_classes = hf_config.num_classes
#             config.return_type = "class_logits"
#         else:
#             config.n_classes = config.d_model
#             config.return_type = "pre_logits"


#         # Video-specific settings
#         if hasattr(hf_config, "tubelet_size"):
#             config.is_video_transformer = True
#             config.video_tubelet_depth = hf_config.tubelet_size[0]
#             config.video_num_frames = hf_config.video_size[0]

#     else:  # TEXT
#         config = HookedTextTransformerConfig()
#         config.d_model = hf_config.hidden_size
#         config.n_layers = hf_config.num_hidden_layers
#         config.n_heads = hf_config.num_attention_heads
#         config.d_head = hf_config.hidden_size // hf_config.num_attention_heads
#         config.d_mlp = hf_config.intermediate_size
#         config.vocab_size = hf_config.vocab_size
#         config.context_length = getattr(hf_config, "max_position_embeddings", 77)
#         config.eps = hf_config.layer_norm_eps

#     return config


def create_config_object(model_name: str, model_type: ModelType) -> ConfigType:
    """
    Create a configuration object for a specific model.

    Args:
        model_name: Model name
        model_type: Model type

    Returns:
        HookedViTConfig or HookedTextTransformerConfig object
    """
    # Get raw configuration dictionary
    try:
        config_dict = MODEL_CONFIGS[model_type][model_name]
    except KeyError:
        raise ValueError(
            f"No configuration found for {model_name} with type {model_type}"
        )

    # Create appropriate config object
    if model_type == ModelType.VISION:
        return HookedViTConfig(**config_dict)
    else:
        return HookedTextTransformerConfig(**config_dict)


def load_original_weights(
    model_name: str,
    category: ModelCategory,
    model_type: ModelType,
    dtype: torch.dtype,
    **kwargs,
) -> Any:
    """
    Load weights for a specific model based on its category.

    Args:
        model_name: Model name
        category: Model category
        model_type: Model type
        dtype: Data type
        **kwargs: Additional loading parameters

    Returns:
        Original weights in source format
    """
    # Handle torch_dtype in kwargs
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]

    # Get local_path from kwargs if it exists
    local_path = kwargs.pop("local_path", None)
    if local_path is not None and category != ModelCategory.OPEN_CLIP:
        raise ValueError("Local path loading is only supported for OpenCLIP models")

    # Special handling for EVA02 models
    if (
        "eva02" in model_name.lower() or "eva_" in model_name.lower()
    ) and category == ModelCategory.OPEN_CLIP:
        return _load_eva02_weights(model_name, **kwargs)

    # Dispatch to appropriate loader based on category
    if category == ModelCategory.TIMM:
        return _load_timm_weights(model_name, **kwargs)

    elif category == ModelCategory.CLIP:
        return _load_clip_weights(model_name, dtype, **kwargs)

    elif category == ModelCategory.OPEN_CLIP:
        return _load_open_clip_weights(model_name, local_path=local_path, **kwargs)

    elif category == ModelCategory.DINO:
        return _load_dino_weights(model_name, dtype, **kwargs)

    elif category == ModelCategory.VIVIT:
        return _load_vivit_weights(model_name, dtype, **kwargs)

    elif category == ModelCategory.VJEPA:
        return _load_vjepa_weights(model_name, **kwargs)

    elif category == ModelCategory.KANDINSKY:
        return _load_kandinsky_weights(model_name, **kwargs)

    else:
        raise ValueError(f"Unsupported model category: {category}")


def convert_weights(
    original_weights: Any,
    model_name: str,
    category: ModelCategory,
    config: ConfigType,
    model_type: ModelType,
) -> Dict[str, torch.Tensor]:
    """
    Convert weights for a specific model.

    Args:
        original_weights: Original weights
        model_name: Model name
        category: Model category
        config: Model configuration
        model_type: Model type

    Returns:
        Converted weights in Prisma format
    """
    # Special case for EVA02 models - use TIMM converter
    if (
        "eva02" in model_name.lower() or "eva_" in model_name.lower()
    ) and category == ModelCategory.OPEN_CLIP:
        return convert_timm_weights(original_weights, config)

    # Special case for CLIP models that need unpacking
    if category == ModelCategory.CLIP and model_type == ModelType.VISION:
        vision_weights = original_weights.vision_model.state_dict()
        projection_weights = original_weights.visual_projection.state_dict()
        return convert_clip_weights(vision_weights, projection_weights, config)

    # Get appropriate converter based on category and type
    if category == ModelCategory.TIMM:
        converter = convert_timm_weights
    elif category == ModelCategory.OPEN_CLIP:
        converter = (
            convert_open_clip_text_weights
            if model_type == ModelType.TEXT
            else convert_open_clip_weights
        )
    elif category == ModelCategory.DINO:
        converter = convert_dino_weights
    elif category == ModelCategory.VIVIT:
        converter = convert_vivet_weights
    elif category == ModelCategory.VJEPA:
        converter = convert_vjepa_weights
    elif category == ModelCategory.KANDINSKY:
        converter = convert_kandinsky_clip_weights
    else:
        raise ValueError(f"No converter available for {category} with {model_type}")

    # Apply converter
    return converter(original_weights, config)


def fill_missing_keys(model, state_dict):
    """
    Fill in any missing keys with default initialization.

    Args:
        model: Model object
        state_dict: State dict with converted weights

    Returns:
        Complete state dict
    """
    # Get the default state dict
    default_state_dict = model.state_dict()

    # Get missing keys
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())

    if missing_keys:
        logging.info(
            f"Filling in {len(missing_keys)} missing keys with default initialization"
        )

    for key in missing_keys:
        if "hf_model" in key:
            # Skip HuggingFace internal keys
            continue

        if "W_" in key:
            logging.warning(f"Missing key for weight matrix: {key}")

        state_dict[key] = default_state_dict[key]

    return state_dict


# ===============================
# Weight Loading Functions
# ===============================


def _load_timm_weights(model_name, **kwargs):
    """Load weights from a TIMM model."""
    try:
        import timm
    except ImportError:
        raise ImportError(
            "TIMM is required but not installed. Install with 'pip install timm'"
        )

    model = timm.create_model(model_name, pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model.state_dict()


def _load_clip_weights(model_name, dtype, **kwargs):
    """Load weights from a CLIP model."""
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_open_clip_weights(model_name, local_path=None, **kwargs):
    """Load weights from an OpenCLIP model."""
    if local_path is not None:
        checkpoint_path = local_path
    else:
        checkpoint_path = download_pretrained_from_hf(
            remove_open_clip_prefix(model_name),
            filename="open_clip_pytorch_model.bin",
        )
    return load_state_dict(checkpoint_path)


def _load_eva02_weights(model_name, **kwargs):
    """Special loader for EVA02 models from TIMM."""
    try:
        import timm
    except ImportError:
        raise ImportError(
            "TIMM is required but not installed. Install with 'pip install timm'"
        )

    model_name_clean = model_name.split("open-clip:")[1]
    name, weights = model_name_clean.split(".")
    name = name.split("/")[1]
    model = timm.create_model(name, pretrained=weights)
    for param in model.parameters():
        param.requires_grad = False
    return model.state_dict()


def _load_dino_weights(model_name, dtype, **kwargs):
    """Load weights from a DINO model."""
    from transformers import ViTModel

    model = ViTModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    for param in model.parameters():
        param.requires_grad = False
    return model.state_dict()


def _load_vivit_weights(model_name, dtype, **kwargs):
    """Load weights from a ViViT model."""
    from transformers import VivitForVideoClassification

    model = VivitForVideoClassification.from_pretrained(
        model_name, torch_dtype=dtype, **kwargs
    )
    for param in model.parameters():
        param.requires_grad = False
    return model.state_dict()


def _load_vjepa_weights(model_name, **kwargs):
    """Load weights from a VJEPA model."""
    try:
        from importlib import resources

        import yaml
        from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAModel
    except ImportError:
        raise ImportError(
            "VJEPA modules not found. Make sure vit_prisma.vjepa_hf is available."
        )

    with resources.open_text("vit_prisma.vjepa_hf", "paths_cw.yaml") as f:
        model_paths = yaml.safe_load(f)
    model_path = model_paths[model_name]["loc"]
    model = VJEPAModel.from_pretrained(model_path)
    return model.state_dict()


def _load_kandinsky_weights(model_name, **kwargs):
    """Load weights from a Kandinsky model."""
    from transformers import CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        cache_dir="/network/scratch/s/sonia.joseph/diffusion",
    ).to("cuda")
    return model.state_dict()


# ===============================
# Helper Functions
# ===============================


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported."""
    return model_name in MODEL_CATEGORIES


def get_supported_model_types(model_name: str) -> list:
    """Get the types supported by a model."""
    if not is_model_supported(model_name):
        return []

    types = [ModelType.VISION]  # All models support vision
    if model_name in TEXT_SUPPORTED_MODELS:
        types.append(ModelType.TEXT)

    return types


def list_available_models(
    category: Optional[ModelCategory] = None,
    model_type: Optional[ModelType] = None,
    detailed: bool = False,
) -> Union[List[str], Dict]:
    """
    List all available models, optionally filtered by category and model type.

    Args:
        category: Optional category filter
        model_type: Optional model type filter
        detailed: Whether to return detailed info (dict) or just names (list)

    Returns:
        List of model names or dictionary with detailed information
    """
    # Import the utilities
    from vit_prisma.model_utils import list_available_models as list_models

    format_type = "tabular" if detailed else "list"
    return list_models(category, model_type, format=format_type)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model details
    """
    from vit_prisma.model_utils import get_model_info as get_info

    return get_info(model_name)
