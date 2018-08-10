# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import threading
import boto3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsValue(object):
    def __init__(self, value):
        self.value = value
        self.timestamp = datetime.utcnow()


class MetricsThread(threading.Thread):
    def __init__(self, interval, target):
        super(MetricsThread, self).__init__()
        self.event = threading.Event()
        self.daemon = True
        self.interval = interval
        self.target = target

    def stop(self):
        self.event.set()

    def run(self):
        while not self.event.wait(self.interval):
            self.target()


class Metrics(object):
    CLOUDWATCH_NAMESPACE = 'benchmarkai-metrics-prod'
    CLOUDWATCH_METRIC_LIMIT = 20
    PUBLISH_INTERVAL = 60.0

    def __init__(self, dimensions=None):
        logger.info('starting metrics agent')
        self.condition = threading.Condition()
        self.metrics = {}
        self.dimensions_list = self.build_dimensions_list(dimensions)
        self.cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        self.publisher = MetricsThread(self.PUBLISH_INTERVAL, self.publish)

    def build_dimensions_list(self, dimensions):
        dimension_list = [{'Name': k, 'Value': v} for k, v in (dimensions or {}).items()]
        logger.debug('dimensions: {}'.format(dimension_list))
        return dimension_list

    def start(self):
        self.publisher.start()

    def stop(self):
        self.publish()
        self.publisher.stop()

    def update(self, metric, value):
        self.condition.acquire()
        if metric not in self.metrics:
            self.metrics[metric] = []

        values = self.metrics[metric]
        values.append(MetricsValue(value))
        self.condition.release()

    def publish(self):
        # publish everything!
        self.condition.acquire()
        metrics = self.metrics
        self.metrics = {}
        self.condition.release()

        data = [self.build_metric_data(k, v) for k, vv in metrics.items() for v in vv]

        for i in range(0, len(data), Metrics.CLOUDWATCH_METRIC_LIMIT):
            chunk = data[i:i + Metrics.CLOUDWATCH_METRIC_LIMIT - 1]
            self.cloudwatch.put_metric_data(Namespace=Metrics.CLOUDWATCH_NAMESPACE, MetricData=chunk)

        logger.info('published {} metrics to cloudwatch'.format(len(data)))
        logger.debug('metrics data: {}'.format(data))

    def build_metric_data(self, metric, value):
        data = {
            'MetricName': metric,
            'Dimensions': self.dimensions_list,
            'Timestamp': value.timestamp,
            'Value': value.value,
            'Unit': self.get_unit(metric)
        }
        return data

    def get_unit(self, metric):
        if '_samples_sec' in metric:
            return 'Count/Second'
        elif '_mb_sec' in metric:
            return 'Megabytes/Second'
        elif '_pct' in metric:
            return 'Percent'
        else:
            return 'None'
