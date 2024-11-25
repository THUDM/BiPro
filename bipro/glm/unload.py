

from subprocess import run, getoutput
from shlex import split
import re

def get_all_nvidia_modules():
    all_modules = getoutput("lsmod").splitlines()
    modules_to_unload = set()
    for m in all_modules:
        m = m.strip()
        m_splitted = re.split("\s+", m)

        module_name = m_splitted[0]
        if len(m_splitted) == 4:
            deps = m_splitted[-1].split(",")
        else:
            deps = []

        if "nvidia" in module_name or any("nvidia" in d for d in deps):
            modules_to_unload.add(module_name)
            for d in deps:
                modules_to_unload.add(d)

    return modules_to_unload

def get_usage_pids(pattern):
    all_files = getoutput("lsof").splitlines()
    pids = set()
    commands = []
    for f in all_files:
        if pattern in f:
            f.strip()
            pid = re.split("\s+", f)[1]
            pids.add(pid)
            commands.append(f)

    return pids



def unload_all_nvidia_modules():
    cnt = 100
    while cnt > 0:
        cnt -= 1
        modules = get_all_nvidia_modules()

        if len(modules) == 0:
            break

        for m in modules:
            pids = get_usage_pids(m)
            for pid in pids:
                run(split(f"killall -9 {pid}"))

            run(split(f"rmmod {m}"))

if __name__ == "__main__":
    get_all_nvidia_modules()
    unload_all_nvidia_modules()