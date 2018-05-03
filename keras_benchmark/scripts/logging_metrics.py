import keras


class LoggingMetrics:
    """Callback that save metrics to a logfile.

    # Arguments
        history_callback: instance of `keras.callbacks.History`.
            Training parameters
            (eg. batch size, number of epochs, loss, acc).
        time_callback: instance of `keras.callbacks.Callback`.
            Training parameters
            (eg. time, time-step, speed).

    # Raises
        TypeError: In case of invalid object instance.
    """

    def __init__(self, history_callback, time_callback):
        self.num_iteration = None
        self.metric_list = []
        self.pattern_list = []
        self.retrieve_metrics(history_callback, time_callback)

    def retrieve_metrics(self, history_callback, time_callback):
        if not isinstance(history_callback, keras.callbacks.History):
            raise TypeError('`history_callback` should be an instance of '
                            '`keras.callbacks.History`')
        if not isinstance(time_callback, keras.callbacks.Callback):
            raise TypeError('`time_callback` should be an instance of '
                            '`keras.callbacks.Callback`')

        if hasattr(history_callback, 'epoch'):
            self.metric_list.append(history_callback.epoch)
            self.pattern_list.append('[Epoch %d]\t')

        if hasattr(time_callback, 'times'):
            self.metric_list.append(time_callback.get_time())
            self.metric_list.append(time_callback.get_time_step())
            self.metric_list.append(time_callback.get_speed())
            self.pattern_list.append('time: %s\t')
            self.pattern_list.append('time_step: %s\t')
            self.pattern_list.append('speed: %s\t')

        if 'loss' in history_callback.history:
            self.metric_list.append(history_callback.history['loss'])
            self.pattern_list.append('train_loss: %.4f\t')

        if 'acc' in history_callback.history:
            self.metric_list.append(history_callback.history['acc'])
            self.pattern_list.append('train_acc: %.4f\t')

        if 'val_loss' in history_callback.history:
            self.metric_list.append(history_callback.history['val_loss'])
            self.pattern_list.append('val_loss: %.4f\t')

        if 'val_acc' in history_callback.history:
            self.metric_list.append(history_callback.history['val_acc'])
            self.pattern_list.append('val_acc: %.4f\t')

        self.num_iteration = history_callback.params['epochs']

    def get_metrics_index(self, idx):
        idx_metric_list = []
        for metric in self.metric_list:
            idx_metric_list.append(metric[idx])
        return tuple(idx_metric_list)

    def save_metrics_to_log(self, logging):
        pattern_str = ''.join(self.pattern_list)
        for i in range(self.num_iteration):
            logging.info(pattern_str % self.get_metrics_index(i))
