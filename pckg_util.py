import os
import subprocess
import sys
import logging

def install_and_import(package, version="", params="", link=""):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        installation_str = package
        installation_cmd_list = ["install"]

        if version:
            installation_str += "==" + version
        installation_cmd_list.append(installation_str)

        if params:
            installation_cmd_list.append(params)

        if link:
            installation_cmd_list.append(link)

        pip.main(installation_cmd_list)
    finally:
        globals()[package] = importlib.import_module(package)


def execute_bash_command(cmd):
    tenv = os.environ.copy()
    tenv['LC_ALL'] = "C"
    bash_command = cmd
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, env=tenv)
    return process.communicate()[0]


def check_gpu_and_torch_compatibility():
    try:
        bash_command = "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,count,utilization.gpu,utilization.memory --format=csv"
        output = execute_bash_command(bash_command).decode()
        print(output)
        if "NVIDIA A100" in output:
            print(True)
            print(sys.executable)
            install_and_import("torch", "1.11.0+cu113", "-f", "https://download.pytorch.org/whl/torch_stable.html")
            install_and_import("torchvision", "0.12.0+cu113", "-f",
                               "https://download.pytorch.org/whl/torch_stable.html")

    except OSError:
        logging.info("GPU device is not available")

