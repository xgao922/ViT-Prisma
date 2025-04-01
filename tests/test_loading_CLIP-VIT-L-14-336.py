import open_clip
import torch

from vit_prisma import load_hooked_model


def test_loading_open_clip_336():
    TOLERANCE = 1e-4
    device = "cpu"

    # Create a fixed random input
    torch.manual_seed(42)
    random_input = torch.randn(1, 3, 336, 336)

    def get_all_layer_outputs(model, input_tensor):
        layer_outputs = []
        layer_names = []

        def hook_fn(module, input, output):
            layer_outputs.append(output)
            layer_names.append(type(module).__name__)

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(input_tensor)

        for hook in hooks:
            hook.remove()

        return layer_outputs, layer_names

    # Load original OpenCLIP model
    og_model_name = "timm/vit_large_patch14_clip_336.openai"
    model_name = "hf-hub:" + og_model_name
    og_model, *_ = open_clip.create_model_and_transforms(model_name)
    og_model.eval()
    all_outputs, layer_names = get_all_layer_outputs(og_model, random_input)

    # Load HookedViT model
    hooked_model = load_hooked_model("open-clip:" + og_model_name)
    hooked_model.to(device)
    hooked_model.eval()

    # Get outputs and cache from hooked model
    final_output_hooked, cache = hooked_model.run_with_cache(random_input)
    final_output_og, *_ = og_model(random_input)

    # Print cache keys for debugging
    print("\nAvailable cache keys:")
    for k in cache:
        print(f"{k}: {cache[k].shape}")

    print("\nLayer-by-layer comparison:")
    # Compare specific layers
    comparisons = [
        (0, "hook_embed", "Embedding output"),
        (1, "hook_full_embed", "Full embedding"),
        (2, "hook_ln_pre", "Pre-LayerNorm"),
        # Add checks for each transformer block
        *[
            (i, f"blocks.{block_idx}.mlp.hook_post", f"Block {block_idx} MLP output")
            for block_idx, i in enumerate(range(8, 118, 10))
        ],
        (118, "blocks.23.mlp.hook_post", "Final block MLP"),
        (124, "ln_final", "Final LayerNorm"),
        (125, "hook_post_head_pre_normalize", "Pre-normalization"),
    ]

    for layer_idx, cache_key, description in comparisons:
        if layer_idx >= len(all_outputs):
            print(f"Warning: Layer index {layer_idx} out of range")
            continue

        og_output = all_outputs[layer_idx]
        if layer_idx == 0:
            og_output = og_output.flatten(2).transpose(1, 2)

        try:
            hooked_output = cache[cache_key]
            max_diff = torch.max(torch.abs(hooked_output - og_output))
            matches = torch.allclose(hooked_output, og_output, atol=TOLERANCE)
            status = "✓" if matches else "✗"
            print(f"{status} {description}: max diff = {max_diff:.2e}")
            assert matches, f"Outputs diverge at {description}!"
        except KeyError:
            print(f"! Missing cache key: {cache_key}")
        except Exception as e:
            print(f"! Error comparing {description}: {str(e)}")

    # Compare final outputs
    max_final_diff = torch.max(torch.abs(final_output_hooked - final_output_og))
    print(f"\nFinal output comparison:")
    print(
        f"Shapes: hooked={final_output_hooked.shape}, original={final_output_og.shape}"
    )
    print(f"Max divergence: {max_final_diff:.2e}")

    assert torch.allclose(
        final_output_hooked, final_output_og, atol=TOLERANCE
    ), f"Final outputs diverge! Max diff: {max_final_diff:.2e}"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_loading_open_clip_336()
