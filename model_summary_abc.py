# ============================================================
#  Project    : Tigers & Goats
#  Module     : Model Summary Utility
#  File       : model_summary_abc.py
#
#  What this file does:
#    Prints a torchsummary report for a MaskablePPO model. It can inspect
#    either a saved .zip model or a fresh model created from the same TnGEnv
#    setup used by training.
#
#  Quick use:
#    1) Summarize a fresh default goat learner model:
#         python model_summary_abc.py
#
#    2) Summarize a saved model:
#         python model_summary_abc.py --model-path artifacts/<experiment>/models/<model>.zip
#
#    3) Summarize a specific policy component:
#         python model_summary_abc.py --module policy.features_extractor
#         python model_summary_abc.py --module policy.mlp_extractor.policy_net
#         python model_summary_abc.py --module policy.mlp_extractor.value_net
#         python model_summary_abc.py --module policy.action_net
#         python model_summary_abc.py --module policy.value_net
#
#    4) Override inferred input size when torchsummary needs help:
#         python model_summary_abc.py --module policy.action_net --input-size 256
#
#    5) Force CPU or CUDA:
#         python model_summary_abc.py --device cpu
#         python model_summary_abc.py --device cuda
#
#    6) Print only the actor/critic architecture report:
#         python model_summary_abc.py --architecture-only
#
#  Fresh-model options:
#    --net-arch 256,256,256
#    --learner-role goat | tiger
#    --tiger-ai tiger_greedy | tiger_smart
#    --goat-opponent-ai goat_random | goat_model
#
#  Notes:
#    - If --model-path is supplied, fresh-model options are ignored.
#    - If a full policy summary fails, the script falls back to component-level
#      summaries for the feature extractor, policy/value MLPs, and heads.
#
# ============================================================

import argparse
from pathlib import Path

import torch
from torch import nn
from torchsummary import summary

from sb3_contrib import MaskablePPO

from env_tng_abc import (
    GOAT_AI_MODEL,
    GOAT_AI_RANDOM,
    GOAT_LEARNER,
    TIGER_AI_GREEDY,
    TIGER_AI_SMART,
    TIGER_LEARNER,
    TnGEnv,
)


def parse_net_arch(raw_value: str) -> list[int]:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("net arch must contain at least one layer size")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "net arch must be a comma-separated list of integers, for example: 256,256,256"
        ) from exc


def parse_input_size(raw_value: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("input size must contain at least one dimension")
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "input size must be a comma-separated list of integers, for example: 25 or 3,84,84"
        ) from exc


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    return device_arg


def build_fresh_model(args, device: str) -> MaskablePPO:
    env = TnGEnv(
        tiger_ai=args.tiger_ai,
        learner_role=args.learner_role,
        goat_opponent_ai=args.goat_opponent_ai,
    )
    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        policy_kwargs=dict(net_arch=args.net_arch),
    )


def load_or_build_model(args, device: str) -> MaskablePPO:
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_file():
            raise SystemExit(f"Model path does not exist: {model_path}")
        return MaskablePPO.load(str(model_path), device=device)
    return build_fresh_model(args, device)


def resolve_module(model: MaskablePPO, module_path: str) -> nn.Module:
    current = model
    for attr in module_path.split("."):
        if not hasattr(current, attr):
            raise SystemExit(f"Could not resolve module path '{module_path}'. Missing attribute: '{attr}'")
        current = getattr(current, attr)

    if not isinstance(current, nn.Module):
        raise SystemExit(f"Resolved object at '{module_path}' is not a torch.nn.Module.")

    return current


def summary_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def format_parameter_count(count: int) -> str:
    return f"{count:,}"


def module_label(name: str, module: nn.Module) -> str:
    total = count_parameters(module)
    trainable = count_parameters(module, trainable_only=True)
    return (
        f"{name}: {module.__class__.__name__} "
        f"(params={format_parameter_count(total)}, trainable={format_parameter_count(trainable)})"
    )


def print_module_tree(module: nn.Module, name: str, max_depth: int, depth: int = 0) -> None:
    indent = "  " * depth
    print(f"{indent}- {module_label(name, module)}")

    if depth >= max_depth:
        children = list(module.named_children())
        if children:
            print(f"{indent}  - ...")
        return

    for child_name, child_module in module.named_children():
        print_module_tree(child_module, child_name, max_depth=max_depth, depth=depth + 1)


def print_named_parameters(module: nn.Module, title: str) -> None:
    print(f"\n=== {title} Parameters ===")
    parameters = list(module.named_parameters())
    if not parameters:
        print("(no parameters)")
        return

    for name, param in parameters:
        grad_flag = "trainable" if param.requires_grad else "frozen"
        print(f"{name}: shape={tuple(param.shape)}, params={format_parameter_count(param.numel())}, {grad_flag}")


def print_policy_architecture(model: MaskablePPO, tree_depth: int) -> None:
    policy = model.policy
    obs_shape = tuple(model.observation_space.shape)
    action_space = model.action_space

    print("\n============================================================")
    print(" MODEL ARCHITECTURE")
    print("============================================================")
    print(f"Policy class: {policy.__class__.__name__}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_space}")
    print(f"Features dim: {getattr(policy, 'features_dim', '(unknown)')}")
    print(f"Total policy params: {format_parameter_count(count_parameters(policy))}")
    print(f"Trainable policy params: {format_parameter_count(count_parameters(policy, trainable_only=True))}")

    print("\n=== Actor / Policy Path ===")
    print("observation")
    print("  -> policy.features_extractor")
    print("  -> policy.mlp_extractor.policy_net")
    print("  -> policy.action_net")
    print("  -> action logits")
    print("  -> SB3 action distribution applies probabilities/masking outside the visible nn.Module tree")

    print("\n=== Critic / Value Path ===")
    print("observation")
    print("  -> policy.features_extractor")
    print("  -> policy.mlp_extractor.value_net")
    print("  -> policy.value_net")
    print("  -> scalar value estimate V(s)")

    print("\n=== Main Components ===")
    print(module_label("policy.features_extractor", policy.features_extractor))
    print(module_label("policy.mlp_extractor.policy_net", policy.mlp_extractor.policy_net))
    print(module_label("policy.mlp_extractor.value_net", policy.mlp_extractor.value_net))
    print(module_label("policy.action_net", policy.action_net))
    print(module_label("policy.value_net", policy.value_net))

    print("\n=== Registered Module Tree ===")
    print_module_tree(policy, "policy", max_depth=tree_depth)

    print_named_parameters(policy, "Policy")

    print("\n=== Softmax / Action Probability Note ===")
    print("MaskablePPO usually exposes policy.action_net as a Linear layer that outputs logits.")
    print("Softmax-style probability conversion is handled by the action distribution, not by an nn.Softmax layer.")
    print("So it is normal if no Softmax module appears in the architecture tree.")


def infer_input_size(model: MaskablePPO, module_path: str, module: nn.Module) -> tuple[int, ...]:
    policy = model.policy
    obs_shape = tuple(model.observation_space.shape)

    if module_path in {"policy", "policy.features_extractor"}:
        return obs_shape

    if module_path in {
        "policy.mlp_extractor",
        "policy.mlp_extractor.policy_net",
        "policy.mlp_extractor.value_net",
    }:
        return (policy.features_dim,)

    if module_path in {"policy.action_net", "policy.value_net"} and hasattr(module, "in_features"):
        return (int(module.in_features),)

    return obs_shape


def print_component_summaries(model: MaskablePPO, device: str) -> None:
    policy = model.policy
    obs_shape = tuple(model.observation_space.shape)
    summary_dev = summary_device(device)

    print("\n=== Features Extractor ===")
    summary(policy.features_extractor, input_size=obs_shape, device=summary_dev)

    print("\n=== Policy MLP ===")
    summary(
        policy.mlp_extractor.policy_net,
        input_size=(policy.features_dim,),
        device=summary_dev,
    )

    print("\n=== Value MLP ===")
    summary(
        policy.mlp_extractor.value_net,
        input_size=(policy.features_dim,),
        device=summary_dev,
    )

    print("\n=== Action Head ===")
    summary(
        policy.action_net,
        input_size=(policy.action_net.in_features,),
        device=summary_dev,
    )

    print("\n=== Value Head ===")
    summary(
        policy.value_net,
        input_size=(policy.value_net.in_features,),
        device=summary_dev,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a torchsummary report for a Tigers and Goats MaskablePPO model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a saved MaskablePPO .zip model. If omitted, a fresh model is created.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="policy",
        help=(
            "Dotted module path to summarize, starting from the loaded MaskablePPO object. "
            "Examples: policy, policy.features_extractor, policy.mlp_extractor.policy_net"
        ),
    )
    parser.add_argument(
        "--input-size",
        type=parse_input_size,
        default=None,
        help=(
            "Optional manual input size for torchsummary, as comma-separated integers. "
            "Examples: 25 or 256. If omitted, the script infers the size for common policy modules."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device to use for the summary pass.",
    )
    parser.add_argument(
        "--net-arch",
        type=parse_net_arch,
        default=[256, 256, 256],
        help="Fresh-model net arch as comma-separated integers. Ignored when --model-path is used.",
    )
    parser.add_argument(
        "--learner-role",
        type=str,
        default=GOAT_LEARNER,
        choices=[GOAT_LEARNER, TIGER_LEARNER],
        help="Learner role used when building a fresh model.",
    )
    parser.add_argument(
        "--tiger-ai",
        type=str,
        default=TIGER_AI_SMART,
        choices=[TIGER_AI_GREEDY, TIGER_AI_SMART],
        help="Tiger opponent used when building a fresh model.",
    )
    parser.add_argument(
        "--goat-opponent-ai",
        type=str,
        default=GOAT_AI_RANDOM,
        choices=[GOAT_AI_RANDOM, GOAT_AI_MODEL],
        help="Goat opponent used when building a fresh model.",
    )
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help="Print the actor/critic architecture report and skip torchsummary.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Maximum depth for the registered module tree in the architecture report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = load_or_build_model(args, device)
    module = resolve_module(model, args.module)

    obs_shape = tuple(model.observation_space.shape)
    action_space = model.action_space
    summary_dev = summary_device(device)
    input_size = args.input_size or infer_input_size(model, args.module, module)

    print(f"Device: {device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_space}")
    print(f"Summary target: {args.module}")
    print(f"Summary input size: {input_size}")

    print_policy_architecture(model, tree_depth=args.tree_depth)

    if args.architecture_only:
        return

    try:
        print("\n============================================================")
        print(" TORCHSUMMARY")
        print("============================================================")
        summary(module, input_size=input_size, device=summary_dev)
    except Exception as exc:
        if args.module != "policy":
            raise
        print(f"\nFull policy summary failed: {exc}")
        print("Falling back to component-level summaries.")
        print_component_summaries(model, device)


if __name__ == "__main__":
    main()
