import platform
import sys

from axiomrl.version import __version__


def print_doctor() -> None:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        metadata = None  # type: ignore[assignment]

    def resolve_version(distribution: str) -> str:
        if metadata is None:
            return "unknown"
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "missing"

    def resolve_first_available_version(*distributions: str) -> str:
        for distribution in distributions:
            version = resolve_version(distribution)
            if version != "missing":
                return version
        return "missing"

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]

    print(f"axiomrl_version={__version__}")
    print(f"python_executable={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch_version={resolve_version('torch')}")
    print(f"gymnasium_version={resolve_version('gymnasium')}")
    print(f"numpy_version={resolve_version('numpy')}")
    print(f"ale_py_version={resolve_first_available_version('ale-py', 'ale_py')}")
    print(f"autorom_version={resolve_first_available_version('AutoROM', 'autorom')}")
    print(f"opencv_python_version={resolve_version('opencv-python')}")
    print(f"pygame_version={resolve_version('pygame')}")
    print(f"minari_version={resolve_version('minari')}")
    try:
        from axiomrl.envs.atari import probe_atari_runtime
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        atari_status = {
            "atari_env_registration": "unavailable",
            "atari_roms_available": False,
            "atari_probe_env_id": "ALE/Tennis-v5",
            "atari_probe_error": str(exc),
        }
    else:
        atari_status = probe_atari_runtime()
    print(f"atari_env_registration={atari_status['atari_env_registration']}")
    print(f"atari_roms_available={atari_status['atari_roms_available']}")
    print(f"atari_probe_env_id={atari_status['atari_probe_env_id']}")
    print(f"atari_probe_error={atari_status.get('atari_probe_error', 'none')}")
    if torch is None:
        print("cuda_available=unknown")
        print("cuda_device_count=unknown")
        print("cuda_device_name=unknown")
        print("torch_cuda_version=unknown")
        return

    cuda_available = torch.cuda.is_available()
    print(f"cuda_available={cuda_available}")
    if not cuda_available:
        print("cuda_device_count=0")
        print("cuda_device_name=none")
    else:
        device_count = torch.cuda.device_count()
        print(f"cuda_device_count={device_count}")
        if device_count < 1:
            print("cuda_device_name=none")
        else:
            try:
                device_name = str(torch.cuda.get_device_name(0))
            except Exception:  # pragma: no cover
                device_name = "unknown"
            print(f"cuda_device_name={device_name}")

    torch_cuda_version = getattr(torch.version, "cuda", None)
    if torch_cuda_version is None:
        print("torch_cuda_version=unknown")
    else:
        print(f"torch_cuda_version={torch_cuda_version}")
