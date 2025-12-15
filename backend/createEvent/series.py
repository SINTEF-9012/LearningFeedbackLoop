import numpy as np
class TimeSeries:
    def __init__(self, channels: int):
        self.channels = channels
        self.data = [[] for _ in range(channels)]
        self.timestamps = []

    def add_data_point(self, timestamp: float, values: list[float]):
        if len(values) != self.channels:
            raise ValueError("Number of values must match number of channels.")
        
        self.timestamps.append(timestamp)

        for i in range(self.channels):
            self.data[i].append(values[i])

    def get_channel(self, channel: int) -> np.ndarray:
        if channel < 0 or channel >= self.channels:
            raise ValueError("Channel index out of range.")
        return np.array(self.data[channel])
    
    def get_timestamps(self) -> np.ndarray:
        return np.array(self.timestamps)
    
