import numpy as np
from dateutil import parser

RAM_LOGS_FILE = './mem_log'
CPU_LOGS_FILE = './cpu_log'
GPU_LOGS_FILE = './nvidiasmi_log'
RAM_LOGS_COMPACT_FILE = './mem_log_comp'
CPU_LOGS_COMPACT_FILE = './cpu_log_comp'
GPU_LOGS_COMPACT_FILE = './nvidiasmi_log_comp'


def compact_ram_logs():
    used_ram = []
    cached_ram = []
    with open(RAM_LOGS_FILE, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        data = line.split(', ')
        used = float(data[-2].split()[0])
        cached = float(data[-1].split()[0])
        used_ram.append(used);
        cached_ram.append(cached)
    with open(RAM_LOGS_COMPACT_FILE, 'w') as f:
        print('Measured in MB', file=f)
        print(
            f'Min RAM usage = {min(used_ram)} | Max RAM usage = {max(used_ram)} | Mean RAM usage = {np.mean(used_ram)} with std = {np.std(used_ram)} ',
            file=f)
        print(
            f'Min RAM cache = {min(cached_ram)} | Max RAM cache = {max(cached_ram)} | Mean RAM cache = {np.mean(cached_ram)} with std = {np.std(cached_ram)} ',
            file=f)


def compact_cpu_logs():
    cpu_us = []
    cpu_sy = []
    cpu_id = []
    with open(CPU_LOGS_FILE, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        data = line.split(', ')
        cpu_us.append(float(data[0].split()[1]))
        cpu_sy.append(float(data[1].split()[0]))
        cpu_id.append(float(data[3].split()[0]))
    cpu_us, cpu_sy, cpu_id = np.array(cpu_us), np.array(cpu_sy), np.array(cpu_id)
    used_cpu = cpu_sy + cpu_us

    with open(CPU_LOGS_COMPACT_FILE, 'w') as f:
        print(
            f'Min idle CPU% = {min(cpu_id)} | Max idle CPU% = {max(cpu_id)} | Mean idle CPU% = {np.mean(cpu_id)} with std = {np.std(cpu_id)} ',
            file=f)
        print(
            f'Min user CPU% = {min(cpu_us)} | Max user CPU% = {max(cpu_us)} | Mean user CPU% = {np.mean(cpu_us)} with std = {np.std(cpu_us)} ',
            file=f)
        print(
            f'Min system CPU% = {min(cpu_sy)} | Max system CPU% = {max(cpu_sy)} | Mean system CPU% = {np.mean(cpu_sy)} with std = {np.std(cpu_sy)} ',
            file=f)
        print(
            f'Min used CPU% = {min(used_cpu)} | Max used CPU% = {max(used_cpu)} | Mean used CPU% = {np.mean(used_cpu)} with std = {np.std(used_cpu)} ',
            file=f)


def compact_gpu_logs():
    timestamps = []
    gpu_utilization = []
    memory_utilization = []
    occupied_memory = []
    with open(GPU_LOGS_FILE, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        data = line.split(', ')
        timestamps.append(data[0])
        occupied_memory.append(float(data[-1].split()[0]))
        memory_utilization.append(float(data[-4].split()[0]))
        gpu_utilization.append(float(data[-5].split()[0]))

    gpu_util, mem_util, occ_mem = np.array(gpu_utilization), np.array(memory_utilization), np.array(occupied_memory)
    start_time, end_time = parser.parse(timestamps[0]), parser.parse(timestamps[-1])
    with open(GPU_LOGS_COMPACT_FILE, 'w') as f:
        print(
            f'Min util of GPU% = {min(gpu_util)} | Max util of GPU% = {max(gpu_util)} | Mean util of GPU% = {np.mean(gpu_util)} with std = {np.std(gpu_util)} ',
            file=f)
        print(
            f'Min util of GPU memory % = {min(mem_util)} | Max util of GPU memory % = {max(mem_util)} | Mean util of GPU memory % = {np.mean(mem_util)} with std = {np.std(mem_util)} ',
            file=f)
        print(
            f'Min occupied GPU memory, MB = {min(occ_mem)} | Max occupied GPU memory, MB = {max(occ_mem)} | Mean occupied GPU memory, MB = {np.mean(occ_mem)} with std = {np.std(occ_mem)} ',
            file=f)
        print(f'Total runtime = {end_time - start_time}', file=f)


if __name__ == '__main__':
    compact_ram_logs()
    compact_cpu_logs()
    compact_gpu_logs()
    print(f'Saved compacted statistics to {RAM_LOGS_COMPACT_FILE}, {RAM_LOGS_COMPACT_FILE}, {GPU_LOGS_COMPACT_FILE}')
    print(f'Algorithm runtime was saved to {GPU_LOGS_COMPACT_FILE}')