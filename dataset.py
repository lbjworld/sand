# coding: utf-8
import logging
import json
import random
import math
from datetime import datetime
from collections import OrderedDict, Counter
import numpy as np
import keras

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MarketDataSet(object):

    SELL_IDX = 0
    BUY_IDX = 1
    HIGH_IDX = 2
    LOW_IDX = 3
    VOL_IDX = 4
    LAST_IDX = 5

    @staticmethod
    def load_data_from_file(file_name, market_type, test_split_rate=0.1):
        with open(file_name, 'r') as f:
            data = f.read()
            records = json.loads(data)
            train_data, test_data = [], []
            total_count = len(records)
            test_count = int(total_count * test_split_rate)
            for idx, r in enumerate(records):
                format_r = MarketDataSet.process_single_record(r)
                if (not format_r) or format_r['type'] != market_type:
                    continue
                if idx < total_count - test_count:
                    train_data.append(format_r.values())
                else:
                    test_data.append(format_r.values())
            logger.info('load data: train({tr}) test({te})'.format(
                tr=len(train_data), te=len(test_data)))
            return train_data, test_data

    @staticmethod
    def process_single_record(record):
        fields = record['fields']
        if not (fields.get('ticker') and fields['ticker'].get('ticker')):
            return None
        res = OrderedDict()
        res['created'] = datetime.strptime(fields['created'], '%Y-%m-%dT%H:%M:%S.%fZ')
        res['type'] = int(fields['market_type'])
        res['sell'] = float(fields['ticker']['ticker']['sell'])
        res['buy'] = float(fields['ticker']['ticker']['buy'])
        res['high'] = float(fields['ticker']['ticker']['high'])
        res['low'] = float(fields['ticker']['ticker']['low'])
        res['vol'] = float(fields['ticker']['ticker']['vol'])
        res['last'] = float(fields['ticker']['ticker']['last'])
        return res

    def sample_data_label(self, data):
        prec = self.profit_rate
        sample_interval = self.sample_interval
        predict_steps = self.predict_steps
        label_steps = self.label_steps
        seed = self.seed
        idx_set = set()
        """
            默认参数的含义：
            每间隔7个step采样一次，采样6次作为一条记录，通过后续的4次采样生成label
        """
        def _mean(idx, records):
            return sum([r[idx] for r in records])/len(records)

        def _min(idx, records):
            return min([r[idx] for r in records])

        def _max(idx, records):
            return max([r[idx] for r in records])

        def mean_last(records):
            return _mean(self.LAST_IDX, records)

        def max_last(records):
            return _max(self.LAST_IDX, records)

        def min_last(records):
            return _min(self.LAST_IDX, records)

        def gen_label(records):
            label_max = max_last(records[-label_steps:])
            label_min = min_last(records[-label_steps:])
            train_m = mean_last(records[:predict_steps])
            label = 0
            if (label_max - train_m) / train_m > prec:
                label = 1
            elif (label_min - train_m) / train_m < -prec:
                label = 2
            # logger.debug('samples: {s}'.format(s=records))
            # logger.debug('label({l}) => {lmax} {lmin} {t}'.format(
            #     l=label, lmax=label_max, lmin=label_min, t=train_m))
            # raw_input()
            return label

        def check_time_series(records):
            """检测采样时序是否正确"""
            time_series = [r[0] for r in records]
            for p1, p2 in zip(time_series + [None], [None] + time_series):
                if p1 and p2:
                    if not (sample_interval*60.0 - 10 <= (p1 - p2).total_seconds() <= sample_interval*60.0 + 120):
                        return False
            return True

        random.seed(seed)
        total_count = len(data)
        padding = (predict_steps + label_steps) * sample_interval
        while len(idx_set) < (total_count - padding):
            idx = random.randint(0, total_count - padding)
            if idx in idx_set:
                continue
            sample_idx = [idx + sample_interval * i for i in range(predict_steps + label_steps)]
            # 加入集合，防止重复选择
            idx_set.add(idx)
            if not check_time_series([data[i] for i in sample_idx]):
                continue
            samples = [data[i][2:] for i in sample_idx]
            label = gen_label(samples)
            train_sample = samples[:predict_steps]
            # logger.debug('generate sample: {s} {l}'.format(s=train_sample, l=label))
            yield train_sample, label
        # 所有的点都已经采样过了
        logger.info('no samples')

    def preprocess_data(self, data, size):
        sampled_data = []
        output_labels = []
        count = 0
        label_stats = Counter()
        while count < size:
            d, l = next(self.sample_data_label(data))
            sampled_data.append(d)
            output_labels.append([l])
            label_stats[l] += 1
            count += 1
        logger.info('data label: {ls}'.format(ls=label_stats))
        one_hot_labels = keras.utils.to_categorical(np.array(output_labels), num_classes=3)
        normalized_data = keras.utils.normalize(np.array(sampled_data), axis=2, order=2)
        # normalized_data = np.array(sampled_data)
        return normalized_data, one_hot_labels

    def load_data(
            self, market_types=[1,2,3], size=10000, profit_rate=0.005, predict_steps=20,
            label_steps=10, sample_interval=5, test_split_rate=0.1, seed=None):
        # init all parameters
        self.market_types = market_types
        self.profit_rate = profit_rate
        self.predict_steps = predict_steps
        self.label_steps = label_steps
        self.sample_interval = sample_interval
        self.seed = seed
        self.size = size
        all_train_data = None
        all_test_data = None
        all_train_labels = None
        all_test_labels = None
        for mtype in market_types:
            train, test = self.load_data_from_file(
                file_name='/code/data/btc_ticker.json',
                market_type=mtype, test_split_rate=test_split_rate)
            sampled_train_data, train_labels = self.preprocess_data(
                train, size=int(size*(1-test_split_rate)))
            sampled_test_data, test_labels = self.preprocess_data(
                test, size=int(size*test_split_rate))
            all_train_data = sampled_train_data if all_train_data is None else np.concatenate(
                (all_train_data, sampled_train_data), axis=0)
            all_train_labels = train_labels if all_train_labels is None else np.concatenate(
                (all_train_labels, train_labels), axis=0)
            all_test_data = sampled_test_data if all_test_data is None else np.concatenate(
                (all_test_data, sampled_test_data), axis=0)
            all_test_labels = test_labels if all_test_labels is None else np.concatenate(
                (all_test_labels, test_labels), axis=0)
        return all_train_data, all_train_labels, all_test_data, all_test_labels
