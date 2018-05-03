""" Utility class for accessing the first epoch time interval.
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/timehistory.py
"""
import keras
import time


class TimeHistory(keras.callbacks.Callback):
    """Callback that extract execution time of every epoch, time-step,
    and speed in terms of sample per sec
    """

    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

    def get_num_samples(self):
        if 'samples' in self.params:
            return self.params['samples']
        elif 'steps' in self.params:
            return self.params['steps']
        else:
            raise ValueError('Incorrect metric parameter')

    def __reformat(self, var):
        if var >= 1:
            var = '%.2f ' % var
            time_format = 'sec'
        elif var >= 1e-3:
            var = '%.2f ' % (var * 1e3)
            time_format = 'msec'
        else:
            var = '%.2f ' % (var * 1e6)
            time_format = 'usec'
        return var, time_format

    def get_time_step(self):
        time_list = []
        num_samples = self.get_num_samples()
        for t in self.times:
            speed = t / num_samples
            speed, time_format = self.__reformat(speed)
            time_list.append(speed + time_format + '/step')
        return time_list

    def get_total_time(self):
        total_time = sum(self.times)
        total_time, time_format = self.__reformat(total_time)
        return total_time + time_format

    def get_time(self):
        time_list = []
        for t in self.times:
            time, time_format = self.__reformat(t)
            time_list.append(time + time_format)
        return time_list

    def get_speed(self):
        samples_list = []
        num_samples = self.get_num_samples()
        for t in self.times:
            sample_sec = num_samples / t
            sample_sec, time_format = self.__reformat(sample_sec)
            samples_list.append(sample_sec + 'samples/' + time_format)
        return samples_list
