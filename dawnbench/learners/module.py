import logging
import mxnet as mx
import time
import os


def get_context(gpu_idxs):
    """

    Parameters
    ----------
    gpu_idxs: None or list of ints
        If None will set context to CPU.
        If list of ints, will set context to given GPUs.

    Returns
    -------

    Context

    """
    if gpu_idxs is None:
        context = [mx.cpu()]
    elif isinstance(gpu_idxs, list):
        context = [mx.gpu(i) for i in gpu_idxs]
    else:
        raise Exception("`gpu_idxs` should be None or list of ints")
    return context


class ModuleLearner():
    def __init__(self, model, run_id, gpu_idxs=None, tensorboard_logging=False):
        """

        Parameters
        ----------
        model: HybridBlock
        gpu_idxs: None or list of ints
            If None will set context to CPU.
            If list of ints, will set context to given GPUs.
        """
        logging.info("Using Module Learner.")
        model.hybridize()
        logging.info("Hybridized model.")
        input = mx.sym.var('data')
        pre_output = model(input)
        output = mx.sym.SoftmaxOutput(pre_output, name='softmax')
        context = get_context(gpu_idxs)
        self.module = mx.mod.Module(symbol=output, context=context,
                                    data_names=['data'], label_names=['softmax_label'])
        self.tensorboard_logging = tensorboard_logging
        if self.tensorboard_logging:
            from mxboard import SummaryWriter
            current_folder = os.path.dirname(os.path.realpath(__file__))
            tensorboard_folder = os.path.join(current_folder, "..", "logs", "tensorboard")
            summary_filepath = os.path.join(tensorboard_folder, run_id)
            self.writer = SummaryWriter(logdir=summary_filepath)


    def fit(self, train_data, valid_data,
            epochs=300,
            lr=None, lr_schedule=None,
            initializer=mx.init.Xavier(),
            optimizer=None,
            kvstore='device',
            log_frequency=10000,
            early_stopping_criteria=None
        ):
        """
        Uses accuracy as training and validation metric.

        Parameters
        ----------
        train_iter : DataIter
            Contains training data
        validation_iter : DataIter
            Contains validation data
        epochs: int
            Number of epochs to run, unless stopped early by early_stopping_criteria.
        lr: float or int
            Learning rate
        lr_schedule : dict
            Contains change points of learning rate.
            Key is the epoch and value is the learning rate.
            Must contain epoch 0.
        initializer : mxnet.initializer.Initializer
        optimizer: mxnet.optimizer.Optimizer
            Defaults to be `mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9)`
        kvstore : str, optional
        log_frequency : int, optional
            Number of batches between logs
        early_stopping_criteria: function (float -> boolean)
            Given validation accuracy, should return True if training should be stopped early.

        Returns
        -------

        None

        Output is logged to file.

        """

        if lr_schedule is None:
            assert lr is not None, "lr must be defined if not using lr_schedule"
            lr_schedule = {0: lr}
        else:
            assert lr is None, "lr should not be defined if using lr_schedule"
            assert 0 in lr_schedule.keys(), "lr for epoch 0 must be defined in lr_schedule"

        mod = self.module
        batch_size = train_data.provide_data[0].shape[0]
        mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
        mod.init_params(initializer=initializer)
        if optimizer is None: optimizer = mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9)
        mod.init_optimizer(kvstore=kvstore, optimizer=optimizer)
        train_metric = mx.metric.create('acc')
        validation_metric = mx.metric.create('acc')
        max_val_acc = {'val_acc': 0, 'trn_acc': 0, 'epoch': 0}

        for epoch in range(epochs):
            epoch_tick = time.time()

            # update learning rate
            if epoch in lr_schedule.keys():
                mod._optimizer.lr = lr_schedule[epoch]
                logging.info("Epoch {}, Changed learning rate.".format(epoch))
            logging.info('Epoch {}, Learning rate={}'.format(epoch, mod._optimizer.lr))
            if self.tensorboard_logging: self.writer.add_scalar(tag='learning_rate', value=mod._optimizer.lr, global_step=epoch + 1)

            train_data.reset()
            train_metric.reset()
            samples_processed = 0
            for batch_idx, batch in enumerate(train_data):
                batch_tick = time.time()
                mod.forward(batch, is_train=True)               # compute predictions
                mod.update_metric(train_metric, batch.label)    # accumulate prediction accuracy
                mod.backward()                                  # compute gradients
                mod.update()                                    # update parameters

                if self.tensorboard_logging:
                    # log to tensorboard (on first batch)
                    if batch_idx == 0:
                        self.writer.add_image(tag="batch", image=batch.data[0], global_step=epoch + 1)

                # log batch speed (if a multiple of log_frequency is contained in the last batch)
                log_batch = (samples_processed // log_frequency) != ((samples_processed + batch_size) // log_frequency)
                if ((batch_idx >= 1) and log_batch):
                    # batch estimate, not averaged over multiple batches
                    speed = batch_size / (time.time() - batch_tick)
                    logging.info('Epoch {}, Batch {}, Speed={:.2f} images/second'.format(epoch, batch_idx, speed))
                samples_processed += batch_size

            # log training accuracy
            _, trn_acc = train_metric.get()
            logging.info('Epoch {}, Training accuracy={}'.format(epoch, trn_acc))
            if self.tensorboard_logging: self.writer.add_scalar(tag='accuracy/training', value=trn_acc*100, global_step=epoch+1)

            # log validation accuracy
            res = mod.score(valid_data, validation_metric)
            _, val_acc = res[0]
            logging.info('Epoch {}, Validation accuracy={}'.format(epoch, val_acc))
            if self.tensorboard_logging: self.writer.add_scalar(tag='accuracy/validation', value=val_acc * 100, global_step=epoch + 1)
            # log maximum validation accuracy
            if val_acc > max_val_acc['val_acc']:
                max_val_acc = {'val_acc': val_acc, 'trn_acc': trn_acc, 'epoch': epoch}
            logging.info(("Epoch {}, Max validation accuracy={} @ "
                          "Epoch {} (with training accuracy={})").format(epoch, max_val_acc['val_acc'],
                                                                         max_val_acc['epoch'], max_val_acc['trn_acc']))

            # log duration of epoch
            logging.info('Epoch {}, Duration={}'.format(epoch, time.time() - epoch_tick))

            if early_stopping_criteria:
                if early_stopping_criteria(val_acc):
                    logging.info("Epoch {}, Reached early stopping target, stopping training.".format(epoch))
                    break


    def save(self, prefix):
        current_folder = os.path.dirname(os.path.realpath(__file__))
        checkpoint_folder = os.path.realpath(os.path.join(current_folder, "..", "logs", "checkpoints"))
        checkpoint_filepath = os.path.join(checkpoint_folder, prefix)
        logging.info("Saved params to " + checkpoint_filepath)
        self.module.save_checkpoint(checkpoint_filepath, epoch=0)
        return checkpoint_filepath


    def load(self, prefix, data_iter):
        current_folder = os.path.dirname(os.path.realpath(__file__))
        checkpoint_folder = os.path.realpath(os.path.join(current_folder, "..", "logs", "checkpoints"))
        checkpoint_filepath = os.path.join(checkpoint_folder, prefix)
        batch_size = data_iter.provide_data[0].shape[0]
        self.module.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label)
        sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_filepath, 0)
        self.module.set_params(arg_params=arg_params, aux_params=aux_params)


    def predict(self,
              test_data,
              log_frequency=10000):
        logging.info('Starting inference.')

        mod = self.module
        batch_size = test_data.provide_data[0].shape[0]
        mod.bind(data_shapes=test_data.provide_data, label_shapes=test_data.provide_label)

        samples_processed = 0
        batch_tick = time.time()
        for pred, batch_idx, batch in mod.iter_predict(test_data):
            pred[0].wait_to_read()
            batch_tock = time.time()
            # log batch speed (if a multiple of log_frequency is contained in the last batch)
            log_batch = (samples_processed // log_frequency) != ((samples_processed + batch_size) // log_frequency)
            warm_up_period = 5
            if ((batch_idx >= warm_up_period) and log_batch):
                # batch estimate, not averaged over multiple batches
                latency = (batch_tock - batch_tick) # seconds
                speed = batch_size / latency
                logging.info('Inference. Batch {}, Latency={:.5f} ms, Speed={:.2f} images/second'.format(batch_idx, latency * 1000, speed))
            samples_processed += batch_size
            batch_tick = time.time()

        logging.info('Completed inference.')