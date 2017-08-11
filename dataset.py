# coding: utf-8
import os
import logging
import json
import random
from datetime import datetime
from collections import OrderedDict, Counter
import numpy as np
import keras

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MarketTickerDataSet(object):

    SELL_IDX = 0
    BUY_IDX = 1
    HIGH_IDX = 2
    LOW_IDX = 3
    VOL_IDX = 4
    LAST_IDX = 5

    @staticmethod
    def load_data_from_file(file_name, market_type):
        with open(file_name, 'r') as f:
            data = f.read()
            records = json.loads(data)
            res_data = []
            for r in records:
                format_r = MarketTickerDataSet.process_single_record(r)
                if (not format_r) or format_r['type'] != market_type:
                    continue
                res_data.append(format_r.values())
            logger.info('load data({tr})'.format(tr=len(res_data)))
            return res_data

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

    def preprocess_data(self, data, size, split_rate):
        train_size = int(size * (1 - split_rate))
        test_size = size - train_size
        train_sampled_data = []
        train_output_labels = []
        count = 0
        train_label_stats = Counter()
        while count < train_size:
            d, l = next(self.sample_data_label(data))
            train_sampled_data.append(d)
            train_output_labels.append([l])
            train_label_stats[l] += 1
            count += 1
        logger.info('train data label: {ls}'.format(ls=train_label_stats))
        train_labels = keras.utils.to_categorical(np.array(train_output_labels), num_classes=3)
        train_normalized_data = keras.utils.normalize(
            np.array(train_sampled_data), axis=2, order=2
        )
        # normalized_data = np.array(sampled_data)
        test_sampled_data = []
        test_output_labels = []
        count = 0
        test_label_stats = Counter()
        while count < test_size:
            d, l = next(self.sample_data_label(data))
            test_sampled_data.append(d)
            test_output_labels.append([l])
            test_label_stats[l] += 1
            count += 1
        logger.info('test data label: {ls}'.format(ls=test_label_stats))
        test_labels = keras.utils.to_categorical(np.array(train_output_labels), num_classes=3)
        test_normalized_data = keras.utils.normalize(
            np.array(train_sampled_data), axis=2, order=2
        )
        return train_normalized_data, train_labels, test_normalized_data, test_labels

    def load_data(
            self, model_name, market_types=[1, 2, 3], size=10000, profit_rate=0.005, predict_steps=20,
            label_steps=10, sample_interval=5, test_split_rate=0.1, seed=None):
        """
            默认参数的含义：
            每间隔5个step采样一次，采样20次作为一条记录，通过后续的10次采样生成label
        """
        # 查看是否有临时数据文件存在
        tmp_data_path = '/code/data/tmp/{mn}.{size}.data'.format(mn=model_name, size=size)
        if os.path.exists(tmp_data_path + '.npz'):
            tmp_data = np.load(tmp_data_path)
            return tmp_data['train_data'], tmp_data['train_labels'], tmp_data['test_data'], tmp_data['test_labels']
        # 重新生成数据文件
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
            origin_data = self.load_data_from_file(
                file_name='/code/data/btc_ticker.json',
                market_type=mtype)
            sampled_train_data, train_labels, sampled_test_data, test_labels = self.preprocess_data(
                origin_data, size=size, split_rate=test_split_rate)
            all_train_data = sampled_train_data if all_train_data is None else np.concatenate(
                (all_train_data, sampled_train_data), axis=0)
            all_train_labels = train_labels if all_train_labels is None else np.concatenate(
                (all_train_labels, train_labels), axis=0)
            all_test_data = sampled_test_data if all_test_data is None else np.concatenate(
                (all_test_data, sampled_test_data), axis=0)
            all_test_labels = test_labels if all_test_labels is None else np.concatenate(
                (all_test_labels, test_labels), axis=0)
        np.savez(
            tmp_data_path, train_data=all_train_data, train_labels=all_train_labels,
            test_data=all_test_data, test_labels=all_test_labels
        )
        return all_train_data, all_train_labels, all_test_data, all_test_labels
