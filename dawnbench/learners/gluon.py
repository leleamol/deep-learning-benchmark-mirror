import logging
import mxnet as mx
import time
import datetime
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


# def accuracy(output, label):
#     output_argmax = output.argmax(axis=1)
#     label_argmax = label
#     equal = output_argmax==label_argmax
#     accuracy = mx.nd.mean(equal).asscalar()
#     return accuracy
#
#
# def evaluate_accuracy(valid_data, model, ctx):
#     acc = 0.
#     count = 0
#     for batch_idx, (data, label) in enumerate(valid_data):
#         data = data.as_in_context(ctx[0])
#         label = label.as_in_context(ctx[0])
#         output = model(data)
#         acc = acc + accuracy(output, label)
#         count += 1
#     return acc / count


class WaitOnReadAccuracy():
    def __init__(self, ctx):
        if isinstance(ctx, list):
            self.ctx = ctx[0]
        else:
            self.ctx = ctx
        self.metric = mx.nd.zeros(1, self.ctx)
        self.num_instance = mx.nd.zeros(1, self.ctx)

    def reset(self):
        self.metric = mx.nd.zeros(1, self.ctx)
        self.num_instance = mx.nd.zeros(1, self.ctx)

    def get(self):
        return float(self.metric.asscalar()) / float(self.num_instance.asscalar())

    def update(self, label, pred):
        # for single context
        if isinstance(label, mx.nd.NDArray) and isinstance(pred, mx.nd.NDArray):
            pred = mx.nd.argmax(pred, axis=1)
            self.metric += (pred == label).sum()
            self.num_instance += label.shape[0]
        # for multi-context where data is partitioned
        elif isinstance(label, list) and isinstance(pred, list):
            for label_part, pred_part in zip(label, pred):
                pred_part = mx.nd.argmax(pred_part, axis=1)
                self.metric += (pred_part == label_part).sum()
                self.num_instance += label_part.shape[0]
        else:
            raise TypeError


def evaluate_accuracy(valid_data, model, ctx):
    if isinstance(ctx, list):
        ctx = ctx[0]
    accuracy = WaitOnReadAccuracy(ctx)
    for batch_idx, (data, label) in enumerate(valid_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(data)
        accuracy.update(label, output)
    return accuracy.get()



class GluonLearner():
    def __init__(self, model, run_id, gpu_idxs=None, hybridize=False, tensorboard_logging=False):
        """

        Parameters
        ----------
        model: HybridBlock
        gpu_idxs: None or list of ints
            If None will set context to CPU.
            If list of ints, will set context to given GPUs.
        """
        logging.info("Using Gluon Learner.")
        self.model = model
        self.run_id = run_id
        if hybridize:
            self.model.hybridize()
            logging.info("Hybridized model.")
        self.context = get_context(gpu_idxs)
        self.tensorboard_logging = tensorboard_logging
        if self.tensorboard_logging:
            from mxboard import SummaryWriter
            current_folder = os.path.dirname(os.path.realpath(__file__))
            tensorboard_folder = os.path.join(current_folder, "..", "logs", "tensorboard")
            summary_filepath = os.path.join(tensorboard_folder, self.run_id)
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
            Number of samples between logs
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

        self.model.initialize(initializer, ctx=self.context)
        if optimizer is None: optimizer = mx.optimizer.SGD(learning_rate=lr_schedule[0], momentum=0.9)
        trainer = mx.gluon.Trainer(params=self.model.collect_params(), optimizer=optimizer, kvstore=kvstore)
        train_metric = mx.metric.Accuracy()
        criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        max_val_acc = {'val_acc': 0, 'trn_acc': 0, 'epoch': 0}

        for epoch in range(epochs):
            epoch_tick = time.time()

            # update learning rate
            if epoch in lr_schedule.keys():
                trainer.set_learning_rate(lr_schedule[epoch])
                logging.info("Epoch {}, Changed learning rate.".format(epoch))
            logging.info('Epoch {}, Learning rate={}'.format(epoch, trainer.learning_rate))
            if self.tensorboard_logging: self.writer.add_scalar(tag='learning_rate', value=trainer.learning_rate, global_step=epoch+1)

            train_metric.reset()
            samples_processed = 0
            for batch_idx, (data, label) in enumerate(train_data):
                batch_tick = time.time()
                batch_size = data.shape[0]

                # partition data across all devices in context
                data = mx.gluon.utils.split_and_load(data, ctx_list=self.context, batch_axis=0)
                label = mx.gluon.utils.split_and_load(label, ctx_list=self.context, batch_axis=0)

                y_pred = []
                losses = []
                with mx.autograd.record():
                    # calculate loss on each partition of data
                    for x_part, y_true_part in zip(data, label):
                        y_pred_part = self.model(x_part)
                        loss = criterion(y_pred_part, y_true_part)
                        # store the losses and do backward after we have done forward on all GPUs.
                        # for better performance on multiple GPUs.
                        losses.append(loss)
                        y_pred.append(y_pred_part)
                    for loss in losses:
                        loss.backward()
                trainer.step(batch_size)
                train_metric.update(label, y_pred)

                if self.tensorboard_logging:
                    # log to tensorboard (on first batch)
                    if batch_idx == 0:
                        self.writer.add_histogram(tag='input', values=x_part, global_step=epoch + 1, bins=100)
                        self.writer.add_histogram(tag='output', values=y_pred_part, global_step=epoch + 1, bins=100)
                        self.writer.add_histogram(tag='loss', values=loss, global_step=epoch + 1, bins=100)
                        self.writer.add_image(tag="batch", image=x_part, global_step=epoch + 1)

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
            val_acc = evaluate_accuracy(valid_data, self.model, ctx=self.context)
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


    def save(self, filename=None):
        current_folder = os.path.dirname(os.path.realpath(__file__))
        checkpoint_folder = os.path.realpath(os.path.join(current_folder, "..", "logs", "checkpoints"))
        if filename:
            checkpoint_filepath = os.path.join(checkpoint_folder, filename)
        else:
            checkpoint_filepath = os.path.join(checkpoint_folder, self.run_id + '.params')
        logging.info("Saved params to " + checkpoint_filepath)
        self.model.save_params(checkpoint_filepath)
        return checkpoint_filepath


    def load(self, filename):
        current_folder = os.path.dirname(os.path.realpath(__file__))
        checkpoint_folder = os.path.realpath(os.path.join(current_folder, "..", "logs", "checkpoints"))
        checkpoint_filepath = os.path.join(checkpoint_folder, filename)
        self.model.load_params(checkpoint_filepath, ctx=self.context)


    def predict(self,
              test_data,
              log_frequency=10000):
        logging.info('Starting inference.')

        samples_processed = 0
        for batch_idx, (data, label) in enumerate(test_data):
            batch_tick = time.time()
            batch_size = data.shape[0]

            # partition data across all devices in context
            data = mx.gluon.utils.split_and_load(data, ctx_list=self.context, batch_axis=0)
            label = mx.gluon.utils.split_and_load(label, ctx_list=self.context, batch_axis=0)

            # calculate loss on each partition of data
            y_pred = []
            for x_part, y_true_part in zip(data, label):
                y_pred_part = self.model(x_part)
                y_pred.append(y_pred_part)

            mx.nd.waitall()
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

        logging.info('Completed inference.')