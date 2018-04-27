import keras
import logging


class LoggingParameter:
    def __init__(self, history_callback, time_callback):
        self.num_iteration = None
        self.parameter_list = []
        self.pattern_list = []
        self.retrieve_params(history_callback, time_callback)

    def retrieve_params(self, history_callback, time_callback):
        if not isinstance(history_callback, keras.callbacks.History):
            raise TypeError('`history_callback` should be an instance of '
                            '`keras.callbacks.History`')
        if not isinstance(time_callback, keras.callbacks.Callback):
            raise TypeError('`time_callback` should be an instance of '
                            '`keras.callbacks.Callback`')

        if hasattr(history_callback, 'epoch'):
            self.parameter_list.append(history_callback.epoch)
            self.pattern_list.append('[Epoch %d]\t')

        if hasattr(time_callback, 'times'):
            self.parameter_list.append(time_callback.get_time())
            self.parameter_list.append(time_callback.get_time_step())
            self.parameter_list.append(time_callback.get_speed())
            self.pattern_list.append('time: %s\t')
            self.pattern_list.append('time_step: %s\t')
            self.pattern_list.append('speed: %s\t')

        if 'loss' in history_callback.history:
            self.parameter_list.append(history_callback.history['loss'])
            self.pattern_list.append('train_loss: %.4f\t')

        if 'acc' in history_callback.history:
            self.parameter_list.append(history_callback.history['acc'])
            self.pattern_list.append('train_acc: %.4f\t')

        if 'val_loss' in history_callback.history:
            self.parameter_list.append(history_callback.history['val_loss'])
            self.pattern_list.append('val_loss: %.4f\t')

        if 'val_acc' in history_callback.history:
            self.parameter_list.append(history_callback.history['val_acc'])
            self.pattern_list.append('val_acc: %.4f\t')

        self.num_iteration = history_callback.params['epochs']

    def get_index_params(self, idx):
        idx_param_list = []
        for param in self.parameter_list:
            idx_param_list.append(param[idx])
        return tuple(idx_param_list)

    def save_params_to_log(self):
        logging.basicConfig(level=logging.INFO)
	#logging.basicConfig(filename='benchmark_cifar10_resnet_log.log', filemode='w',
        #            format='%(message)s', level=logging.INFO)
        pattern_str = ''.join(self.pattern_list)
        for i in range(self.num_iteration):
            logging.info(pattern_str % self.get_index_params(i))
