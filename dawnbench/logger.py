import logging
import os
import sys
import datetime


def construct_run_id(filename):
    """

    Parameters
    ----------
    filename : str
        Should pass __file__ from the experiment file.

    Returns
    -------
    run_id : str
        Can be used for log filename and TensorBoard files.
    """
    experiment = os.path.splitext(os.path.basename(filename))[0]
    datetime_str = str(datetime.datetime.now()).replace(' ', '-')[:19]
    run_id = experiment + '_' + datetime_str
    return run_id


def configure_root_logger(run_id):
    """
    Sets the root logger to log to file and stdout, with correct formatting.

    Parameters
    ----------
    run_id : str

    Returns
    -------
    None
    """
    current_folder = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(current_folder, "logs", "text")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_filename = run_id + '.log'
    log_filepath = os.path.join(log_folder, log_filename)
    print("Writing logs to {}".format(log_filepath))
    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(log_filepath)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


