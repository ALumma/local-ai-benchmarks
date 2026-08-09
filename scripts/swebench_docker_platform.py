from __future__ import annotations

from typing import Any


UBUNTU_BASE_IMAGE = "ubuntu:22.04"


def docker_platform_for_arch(arch: str) -> str:
    if arch == "arm64":
        return "linux/arm64/v8"
    if arch == "x86_64":
        return "linux/amd64"
    raise ValueError(f"Unsupported Docker architecture: {arch}")


def docker_image_arch_for_arch(arch: str) -> str:
    if arch == "arm64":
        return "arm64"
    if arch == "x86_64":
        return "amd64"
    raise ValueError(f"Unsupported Docker architecture: {arch}")


def ensure_ubuntu_base_image(arch: str, client: Any | None = None) -> None:
    import docker

    docker_client = client or docker.from_env()
    expected_arch = docker_image_arch_for_arch(arch)
    platform = docker_platform_for_arch(arch)

    try:
        image = docker_client.images.get(UBUNTU_BASE_IMAGE)
        actual_arch = image.attrs.get("Architecture")
    except docker.errors.ImageNotFound:
        actual_arch = None

    if actual_arch == expected_arch:
        return

    if actual_arch:
        print(
            f"Replacing cached {UBUNTU_BASE_IMAGE} ({actual_arch}) with {platform}."
        )
    else:
        print(f"Pulling {UBUNTU_BASE_IMAGE} for {platform}.")

    image = docker_client.images.pull(UBUNTU_BASE_IMAGE, platform=platform)
    pulled_arch = image.attrs.get("Architecture")
    if pulled_arch != expected_arch:
        raise RuntimeError(
            f"Docker pulled {UBUNTU_BASE_IMAGE} with architecture {pulled_arch!r}; "
            f"expected {expected_arch!r}."
        )
