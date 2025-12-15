import numpy as np
from series import TimeSeries

class TurnmillProcess:
    def __init__(self, data_dict: dict, nChannels: int):
        self.nChannels = nChannels
        self.d = data_dict["d"] # diameter of tool [mm]
        self.z = data_dict["z"] #number of teeth 
        self.ap = data_dict["ap"] # axial depth of cut [mm]
        self.ae = data_dict["ae"] # radial depth of cut [mm]
        self.n = data_dict["n"] # spindle speed [rpm]
        self.f = data_dict["f"] # feed per tooth [mm/tooth]
        self.fg = self.n/60 # spindle frequency
        self.fp = self.fg * self.z #tooth passing frequency
        self.type = data_dict["type"] #up or down milling

        self.series = TimeSeries(nChannels)

    def add_datapoint(self, timestamp: float, values: list[float]):
        self.series.add_data_point(timestamp, values)

    def get_channel_array(self, channel: int) -> np.ndarray:
        return self.series.get_channel(channel)
    
    def get_timestamps(self) -> np.ndarray:
        return self.series.get_timestamps()