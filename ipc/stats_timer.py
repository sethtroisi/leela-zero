import time
from collections import deque


class StatsTimer():
    def __init__(self, name, samples = 5, sample_freq = 100):
        self.name = name
        self.sum = 0
        self.count = 0

        self.samples = deque(maxlen = samples)
        self.sample_freq = sample_freq

        self.t = None


    def start(self):
        assert self.t == None
        self.t = time.time()


    def stop(self):
        assert self.t
        t = time.time() - self.t
        self.t = None
        self.sum += t
        self.count += 1
        if self.count % self.sample_freq == 0:
            self.samples.appendleft(t)


    def getSummary(self):
        data_summary = "{}: {:.1f}s = {}".format(
            self.name, self.sum, StatsTimer._timePer(self.sum, self.count))
        sample_summary = self._prettyPrintSamples()
        return data_summary + sample_summary

    def _prettyPrintSamples(self):
        if len(self.samples) == 0: return ""

        # assume sample times are all milliseconds
        times = map(lambda s: "{:.1f} ms".format(s * 1e3), self.samples)
        iter_time = StatsTimer._timePer(sum(self.samples), len(self.samples))
        return ", samples [{}] => {}".format(", ".join(times), iter_time)

    @staticmethod
    def _formatTime(t):
        if t < 0.001:
            return "{:.1f} micros".format(t * 1e6)
        elif t < 1:
            return "{:.1f} millis".format(t * 1e3)
        else:
            return "{:.1f} seconds".format(t)

    @staticmethod
    def _timePer(time, count):
        if count > 0:
            time_per = StatsTimer._formatTime(time / count)
        else:
            time_per = "???"
        return time_per + "/iter"
