import numpy as np


class Resampler:
    def __init__(self, orig_sr, target_sr):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.ratio = target_sr / orig_sr
        self._index_array = None
        self._int_index = None
        self._frac_index = None

    def resample(self, audio):
        if self.orig_sr == self.target_sr:
            return audio
        output_length = int(len(audio) * self.ratio)
        if self._index_array is None or len(self._index_array) != output_length:
            self._index_array = np.arange(output_length) / self.ratio
            self._int_index = self._index_array.astype(int)
            self._frac_index = self._index_array - self._int_index
        int_index = np.minimum(self._int_index, len(audio) - 1)
        next_index = np.minimum(int_index + 1, len(audio) - 1)
        return audio[int_index] * (1 - self._frac_index) + audio[next_index] * self._frac_index
