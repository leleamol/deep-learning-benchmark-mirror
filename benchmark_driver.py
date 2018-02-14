from __future__ import print_function
import argparse
import os
import subprocess
import re
import json
from ast import literal_eval
import logging
logging.basicConfig(level=logging.INFO)

try:
    import ConfigParser
    config = ConfigParser.ConfigParser()
except ImportError:
    import configparser
    config = configparser.ConfigParser()

from utils import cpu_gpu_profiler, cfg_process

# TODO add detailed error/exception handling in the script
from utils.errors import MetricComputeMethodError, MetricPatternError


CONFIG_TEMPLATE_DIR = './task_config_template.cfg'
CONFIG_DIR = './task_config.cfg'
RESULT_FILE_PATH = './dlbenchmark_result.json'
NUMERIC_PATTERN = "(\d+\.\d+|\d+)"


class BenchmarkMetricComputeMethod:
    @staticmethod
    def compute(metric_compute_method, metric):
        if metric_compute_method == 'average':
            return 1.0 * sum(metric) / len(metric)
        elif metric_compute_method == 'last':
            return metric[-1]
        elif metric_compute_method == 'total':
            return sum(metric)
        else:
            raise MetricComputeMethodError("This metric compute method is not supported!")


class BenchmarkResultManager(object):
    def __init__(self, log_file_location, metric_patterns, metric_names, metric_compute_methods):
        """ Manages holding the map of the result data.

        :param log_file_location: string
            file location
        :param metric_patterns: list
            list of metric patterns
        :param metric_names: list
            list of metric names, in the same order as metric patterns
        :param metric_compute_methods: list
            list of metric computation method, in the same order as metric patterns
        """
        self.metric_map = {}
        if not os.path.isfile(log_file_location):
            raise Exception("log file was missing!")
        with open(log_file_location, 'rb') as f:
            self.log_file = f.read()
        assert isinstance(metric_patterns, list), "metric_patterns is expected to be a list."
        assert isinstance(metric_names, list), "metric_names is expected to be a list."
        assert isinstance(metric_compute_methods, list), "metric_compute_methods is expected to be a list."
        assert len(metric_patterns) == len(metric_names) == len(metric_compute_methods),\
            "metric_patterns, metric_names, metric_compute_methods should have same length."
        self.metric_patterns = metric_patterns
        self.metric_names = metric_names
        self.metric_compute_methods = metric_compute_methods

    @staticmethod
    def __get_float_number(s):
        matches = re.findall(NUMERIC_PATTERN, s)
        if len(matches) == 1:
            return eval(re.findall(NUMERIC_PATTERN, s)[0])
        else:
            raise MetricPatternError("Can not find number in the located metric pattern.")

    def parse_log(self):
        for i in range(len(metric_patterns)):
            pattern = self.metric_patterns[i]
            name = self.metric_names[i]
            metric = re.findall(pattern, self.log_file)
            if len(metric) == 0:
                raise MetricPatternError("Can not locate provided metric pattern.")
            metric = map(self.__get_float_number, metric)
            metric_result = BenchmarkMetricComputeMethod.compute(
                metric_compute_method=self.metric_compute_methods[i],
                metric=metric
            )
            self.metric_map[name] = metric_result

    def save_to(self, result_file_location):
        if os.path.isfile(result_file_location):
            os.remove(result_file_location)
        with open(result_file_location, 'w') as f:
            f.write(json.dumps(self.metric_map))


def benchmark(command_to_execute, metric_patterns,
              metric_names, metric_compute_methods,
              num_gpus, task_name, suffix, framework):
    """ Benchmark Driver Function

    :param command_to_execute: string
        The command line to execute the benchmark job
    :param metric_patterns: list
        list of metric patterns
    :param metric_names: list
        list of metric names, in the same order as metric patterns
    :param metric_compute_methods: list
        list of metric computation method, in the same order as metric patterns
    :param num_gpus: int
        number of gpus to use for training
    :param task_name: str
        task name
    :param suffix: str
        metric suffix in the output
    :param framework: str
        name of the framework
    :return:
    """
    log_file_location = task_name + ".log"
    log_file = open(log_file_location, 'w')
    logging.info("Executing Command: %s" % command_to_execute)

    cpu_gpu_memory_usage = {}
    process = subprocess.Popen(
            command_to_execute,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    # when num_gpus == 0, the cpu_gpu_profiler will only profile cpu usage
    with cpu_gpu_profiler.Profiler(cpu_gpu_memory_usage, num_gpus, process.pid):
        process.communicate()
    log_file.close()

    result = BenchmarkResultManager(
        log_file_location=log_file_location,
        metric_patterns=metric_patterns,
        metric_names=metric_names,
        metric_compute_methods=metric_compute_methods,
    )

    result.metric_map.update(cpu_gpu_memory_usage)
    result.parse_log()

    # prepend task name and append suffix if any
    update_metric_map = {}
    for metric in result.metric_map:
        map_key = task_name + "." + metric
        if suffix:
            map_key += "." + suffix
        if framework:
            map_key = framework + "." + map_key
        update_metric_map[map_key] = result.metric_map[metric]
    logging.info(update_metric_map)
    result.metric_map = update_metric_map
    result.save_to(RESULT_FILE_PATH)
    # clean up
    #os.remove(log_file_location)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a benchmark task.")
    parser.add_argument('--framework', type=str, help='Framework name e.g. mxnet')
    parser.add_argument('--task-name', type=str, help='Task Name e.g. resnet50_cifar10_symbolic.')
    parser.add_argument('--num-gpus', type=int, help='Numbers of gpus. e.g. --num-gpus 8')
    parser.add_argument('--epochs', type=int, help='Numbers of epochs for training. e.g. --epochs 20')
    parser.add_argument('--metrics-suffix', type=str, help='Metrics suffix e.g. --metrics-suffix daily')
    parser.add_argument('--kvstore', type=str, default='device',help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',help='floating point precision to use')
      
    
    args = parser.parse_args()

    # modify the template config file and generate the user defined config file.
    cfg_process.generate_cfg(CONFIG_TEMPLATE_DIR, CONFIG_DIR, **vars(args))
    config.read(CONFIG_DIR)

    # the user defined config file should only have one task
    selected_task = config.sections()[0]
    metric_patterns = literal_eval(config.get(selected_task, "patterns"))
    metric_names = literal_eval(config.get(selected_task, "metrics"))
    metric_compute_methods = literal_eval(config.get(selected_task, "compute_method"))
    command_to_execute = config.get(selected_task, "command_to_execute")
    num_gpus = int(config.get(selected_task, "num_gpus"))

    benchmark(
        command_to_execute=command_to_execute,
        metric_patterns=metric_patterns,
        metric_names=metric_names,
        metric_compute_methods=metric_compute_methods,
        num_gpus=num_gpus,
        task_name=selected_task,
        suffix=args.metrics_suffix,
        framework=args.framework
    )

    # clean up
    os.remove(CONFIG_DIR)
