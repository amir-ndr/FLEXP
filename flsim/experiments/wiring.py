"""
wiring.py: Config loading and component factory functions.

Shared by experiments/base.py and any other entry-point module.
Contains no main() and no direct simulation logic — only pure factory
functions that map config values to concrete objects.

To register a new algorithm, allocator, or channel model, add an elif
branch to the relevant _make_* function here.
"""

import copy
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from flsim.allocators.equal_split import EqualSplitAllocator
from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedprox import FedProx
from flsim.channel.exp_fading import ExpFadingChannelModel
from flsim.channel.fdma import FDMAChannelModel
from flsim.data.dirichlet import DirichletPartitioner
from flsim.data.iid import IIDPartitioner
from flsim.data.shard import ShardPartitioner
from flsim.data.loaders.mnist import load_mnist
from flsim.data.loaders.cifar10 import load_cifar10
from flsim.data.loaders.cifar100 import load_cifar100
from flsim.data.loaders.ham10000 import load_ham10000
from flsim.profiles.factory import create_client_profiles


_BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> SimpleNamespace:
    """
    Load an experiment YAML and deep-merge it over base.yaml.

    Args:
        config_path (str): path to the experiment-specific YAML.

    Returns:
        SimpleNamespace: merged config with attribute access.
    """
    with open(_BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    merged = _deep_merge(base, override)
    return _dict_to_ns(merged)


def set_seeds(seed: int) -> None:
    """Set numpy and PyTorch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Component factories — add new implementations here
# ---------------------------------------------------------------------------

def _make_partitioner(cfg_data):
    if cfg_data.partition == "iid":
        return IIDPartitioner()
    elif cfg_data.partition == "shard":
        return ShardPartitioner(
            num_shards=cfg_data.num_shards,
            shards_per_client=cfg_data.shards_per_client,
        )
    elif cfg_data.partition == "dirichlet":
        return DirichletPartitioner(alpha=cfg_data.dirichlet_alpha)
    raise ValueError(f"Unknown partition: '{cfg_data.partition}'")


def _make_algorithm(name: str):
    """
    Map algorithm name (from YAML) to a FederatedAlgorithm instance.

    Add new algorithms here:
        if name == "scaffold": return SCAFFOLD()
    """
    if name == "fedavg":
        return FedAvg()
    if name == "fedprox":
        return FedProx(mu=0.01)   # default mu; override via components= in experiments
    raise ValueError(f"Unknown algorithm: '{name}'. Register it in wiring._make_algorithm().")


def _make_allocator(config):
    """
    Map allocator name (from config) to a ResourceAllocator instance.

    Add new allocators here:
        if name == "channel_proportional": return ChannelProportionalBW()
    """
    # Only EqualSplitAllocator today — extend with elif branches.
    return EqualSplitAllocator()


def _make_channel_model(config, noise_psd_w_per_hz: float):
    """
    Instantiate the channel model specified by wireless.channel_model.

    Add new channel models here:
        if name == "my_model": return MyChannelModel(...)
    """
    cfg_w = config.wireless
    name  = cfg_w.channel_model
    if name == "path_loss":
        return FDMAChannelModel(
            total_bandwidth_hz=cfg_w.total_bandwidth_hz,
            noise_psd_w_per_hz=noise_psd_w_per_hz,
            min_distance_m=cfg_w.min_distance_m,
        )
    elif name == "exp_fading":
        return ExpFadingChannelModel(
            h0=cfg_w.h0_path_loss_constant,
            total_bandwidth_hz=cfg_w.total_bandwidth_hz,
            noise_psd_w_per_hz=noise_psd_w_per_hz,
            min_distance_m=cfg_w.min_distance_m,
            path_loss_exponent=getattr(cfg_w, "exp_fading_path_exponent", 2.0),
        )
    raise ValueError(
        f"Unknown channel_model '{name}'. Choose 'path_loss' or 'exp_fading', "
        f"or register a new one in wiring._make_channel_model()."
    )


def _make_profiles(config, num_samples_list: list, rng):
    """Create one ClientSystemProfile per client from system + wireless config."""
    cfg_s = config.system
    cfg_w = config.wireless
    return create_client_profiles(
        num_clients=config.data.num_clients,
        num_samples_list=num_samples_list,
        tx_power_dbm=cfg_w.tx_power_dbm,
        min_distance_m=cfg_w.min_distance_m,
        rng=rng,
        deployment_shape=cfg_w.deployment_shape,
        area_side_m=getattr(cfg_w, "area_side_m", 500.0),
        area_radius_m=getattr(cfg_w, "area_radius_m", 500.0),
        dist_min_m=getattr(cfg_w, "dist_min_m", 100.0),
        dist_max_m=getattr(cfg_w, "dist_max_m", 1000.0),
        cpu_freq_mode=cfg_s.cpu_freq_mode,
        cpu_frequency_hz=getattr(cfg_s, "cpu_frequency_hz", 2.0e9),
        cpu_freq_min_ghz=getattr(cfg_s, "cpu_freq_min_ghz", 0.1),
        cpu_freq_max_ghz=getattr(cfg_s, "cpu_freq_max_ghz", 0.8),
        cpu_freq_step_ghz=getattr(cfg_s, "cpu_freq_step_ghz", 0.1),
        tx_power_w_min=getattr(cfg_w, "tx_power_w_min", None),
        tx_power_w_max=getattr(cfg_w, "tx_power_w_max", None),
        cycles_per_sample_min=cfg_s.cycles_per_sample_min,
        cycles_per_sample_max=cfg_s.cycles_per_sample_max,
        shadowing_std_db=getattr(cfg_w, "shadowing_std_db", 0.0),
    )


def _model_name_for_dataset(dataset: str, model_name: str = None) -> str:
    """
    Resolve which model architecture to build.

    If model_name is given (data.model_name in YAML, e.g. "resnet18",
    "vgg16", "alexnet" — see flsim.models.factory.list_models() for the full
    registry), it is used directly, letting you pick any registered CIFAR-10
    architecture without touching this mapping. Otherwise falls back to each
    dataset's default model.
    """
    if model_name is not None:
        return model_name
    mapping = {
        "mnist": "mnist_cnn",
        "cifar10": "cifar_cnn",
        "cifar100": "cifar_cnn",
        "ham10000": "cifar_cnn",
    }
    if dataset not in mapping:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(mapping)}")
    return mapping[dataset]


def _num_classes_for_dataset(dataset: str, num_classes: int = None) -> int:
    """
    Resolve how many output classes the model's final layer needs.

    If num_classes is given (data.num_classes in YAML), it is used directly.
    Otherwise falls back to each dataset's natural class count.
    """
    if num_classes is not None:
        return num_classes
    mapping = {"mnist": 10, "cifar10": 10, "cifar100": 100, "ham10000": 7}
    if dataset not in mapping:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(mapping)}")
    return mapping[dataset]


def _load_dataset(config):
    """
    Load (train_dataset, test_dataset) for config.data.dataset.

    Centralizes the dataset dispatch shared by every experiment base class
    (Experiment, AsyncExperiment, SplitExperiment) — add a new dataset here
    once and every paradigm picks it up.

    ham10000 additionally requires config.data.ham10000_root (no
    auto-download is available for it — see flsim.data.loaders.ham10000).
    """
    dataset = config.data.dataset
    if dataset == "mnist":
        return load_mnist()
    elif dataset == "cifar10":
        return load_cifar10()
    elif dataset == "cifar100":
        return load_cifar100()
    elif dataset == "ham10000":
        root = getattr(config.data, "ham10000_root", None)
        if not root:
            raise ValueError(
                "data.ham10000_root must be set to load HAM10000 (no "
                "auto-download available — point it at the folder containing "
                "HAM10000_metadata.csv and the image subfolder(s); see "
                "flsim.data.loaders.ham10000 module docstring)."
            )
        return load_ham10000(root=os.path.expanduser(root))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns
