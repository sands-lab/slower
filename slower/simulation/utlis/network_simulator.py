import time
import random
import sys
from slower.common import BatchData


class NetworkSimulator:
    def __init__(self, avg_latency_ms, latency_variance_ms, avg_bandwidth_mbps, bandwidth_variance_mbps):
        self.avg_latency_ms = avg_latency_ms
        self.latency_variance_ms = latency_variance_ms
        self.avg_bandwidth_mbps = avg_bandwidth_mbps
        self.bandwidth_variance_mbps = bandwidth_variance_mbps

    def simulate_latency(self):
        latency = random.gauss(self.avg_latency_ms, self.latency_variance_ms)
        time.sleep(max(0, latency) / 1000)

    def simulate_bandwidth(self, data_size):
        bandwidth_mbps = max(0, random.gauss(self.avg_bandwidth_mbps, self.bandwidth_variance_mbps))
        if bandwidth_mbps > 0:
            transfer_time = data_size / (bandwidth_mbps * 1024 * 1024 / 8)
            time.sleep(transfer_time)

    def simulate_network(self, batch_data: BatchData):
        data_size = get_data_size(batch_data)
        self.simulate_latency()
        self.simulate_bandwidth(data_size)


def get_data_size(batch_data) -> int:
    def get_size(obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(vars(obj), seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size

    return get_size(batch_data)
