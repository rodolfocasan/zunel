# setup.py
import re
import sys
import platform
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop





TORCH_VERSION = 'torch>=2.0.0'

CUDA_TO_INDEX = {
    (12, 6): 'https://download.pytorch.org/whl/cu126',
    (12, 7): 'https://download.pytorch.org/whl/cu126',
    (12, 8): 'https://download.pytorch.org/whl/cu128',
    (12, 9): 'https://download.pytorch.org/whl/cu128',
    (13, 0): 'https://download.pytorch.org/whl/cu130',
}

CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
ROCM_INDEX = 'https://download.pytorch.org/whl/rocm7.1'


def _torch_installed():
    result = subprocess.run(
        [sys.executable, '-c', 'import torch'],
        capture_output=True
    )
    return result.returncode == 0


def _detect_cuda():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _detect_rocm():
    result = subprocess.run(['rocm-smi', '--showdriverversion'], capture_output=True, text=True)
    return result.returncode == 0


def _is_aarch64():
    return platform.machine().lower() in ('aarch64', 'arm64')


def _resolve_index():
    cuda = _detect_cuda()

    if cuda is not None:
        major, minor = cuda
        index_url = CUDA_TO_INDEX.get((major, minor))

        if index_url is None:
            same_major = [(m, n) for (m, n) in CUDA_TO_INDEX if m == major]
            if same_major:
                closest = max(same_major, key=lambda v: v[1])
                index_url = CUDA_TO_INDEX[closest]
            else:
                index_url = CUDA_TO_INDEX[(12, 8)]

        if (major, minor) == (12, 8) and _is_aarch64():
            print('[zunel] CUDA 12.8 on aarch64: using default PyPI index')
            return None

        print(f'[zunel] Detected CUDA {major}.{minor} -> {index_url}')
        return index_url

    if _detect_rocm():
        print(f'[zunel] Detected ROCm -> {ROCM_INDEX}')
        return ROCM_INDEX

    print(f'[zunel] No GPU detected -> {CPU_INDEX}')
    return CPU_INDEX


def install_torch():
    if _torch_installed():
        print('[zunel] PyTorch already installed, skipping.')
        return

    index_url = _resolve_index()
    cmd = [sys.executable, '-m', 'pip', 'install', TORCH_VERSION]

    if index_url:
        cmd += ['--index-url', index_url]

    print(f'[zunel] Installing PyTorch: {" ".join(cmd)}')
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print('[zunel] ERROR: PyTorch installation failed.')
        sys.exit(1)





class ZunelInstall(install):
    def run(self):
        install_torch()
        super().run()


class ZunelDevelop(develop):
    def run(self):
        install_torch()
        super().run()





setup(
    name = 'zunel',
    version = '1.0.5',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'soundfile',
        'librosa',
        'gradio',
    ],
    cmdclass = {
        'install': ZunelInstall,
        'develop': ZunelDevelop,
    },
    python_requires = '>=3.8',
)