import importlib.metadata
import logging
import os
import subprocess
import sys


def install_and_import(package, version="", params="", link="", packageimportname=""):
    try:
        if importlib.metadata.version(package) != version:
            raise ImportError
        importlib.import_module(package)
    except ImportError:
        pass

        installation_str = package
        installation_cmd_list = ["install"]

        if version:
            installation_str += "==" + version
        installation_cmd_list.append(installation_str)

        if params:
            installation_cmd_list.append(params)

        if link:
            installation_cmd_list.append(link)

        subprocess.check_call(
            [sys.executable, "-m", "pip", *installation_cmd_list]
        )
    finally:
        if packageimportname is None or packageimportname == "":
            globals()[package] = importlib.import_module(package)
        else:
            globals()[packageimportname] = importlib.import_module(packageimportname)


def execute_bash_command(cmd):
    tenv = os.environ.copy()
    tenv["LC_ALL"] = "C"
    bash_command = cmd
    process = subprocess.Popen(
        bash_command.split(), stdout=subprocess.PIPE, env=tenv
    )
    return process.communicate()[0]


def check_gpu_and_torch_compatibility():
    try:
        import platform

        if platform.system() == "Windows":
            install_and_import(
                "torch",
                "1.12.1+cu116",
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            )
        else:
            bash_command = "nvidia-smi --query-gpu=name --format=csv"
            output = execute_bash_command(bash_command).decode()
            if "NVIDIA A100" in output:
                install_and_import(
                    "torch",
                    "1.11.0+cu113",
                    "-f",
                    "https://download.pytorch.org/whl/torch_stable.html",
                )
                install_and_import(
                    "torchvision",
                    "0.12.0+cu113",
                    "-f",
                    "https://download.pytorch.org/whl/torch_stable.html",
                )
    except OSError:
        logging.info("GPU device is not available")


def install_mmcv():
    import torch
    torch_version = str(torch.__version__)[:4]
    cuda_version = str(torch.version.cuda).replace(".", "")
    install_and_import("mmcv",
                       version="2.1.0",
                       params="-f",
                       link=f"https://download.openmmlab.com/mmcv/dist/cu{cuda_version}/torch{torch_version}/index.html")