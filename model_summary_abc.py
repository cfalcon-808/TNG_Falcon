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

    try:
        summary(module, input_size=input_size, device=summary_dev)
    except Exception as exc:
        if args.module != "policy":
            raise
        print(f"\nFull policy summary failed: {exc}")
        print("Falling back to component-level summaries.")
        print_component_summaries(model, device)


if __name__ == "__main__":
    main()
