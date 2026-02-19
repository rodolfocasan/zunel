# zunel/utils/resources.py
import os
import math





def get_cpu_count_by_percentage(percentage: float) -> int:
    if not (0 < percentage <= 100):
        raise ValueError("[zunel (cpu count)] The percentage must be between 0 and 100.")

    total_cpus = os.cpu_count() or 1
    #cpu_count = math.floor((percentage / 100) * total_cpus)
    cpu_count = math.ceil((percentage / 100) * total_cpus)
    return max(1, cpu_count)