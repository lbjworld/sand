# coding: utf-8
import logging
import keras
import time
import numpy as np

from utils import find_latest_model
from dataset import MarketTickerDataSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = './models'
MODEL_NAME = 'weighted_price_box_l3'


class EpochCheckPoint(keras.callbacks.Callback):
    def __init__(self, path, model_name, checkpoint_per_epochs=100):
        self.save_path = path
        self.model_name = model_name
        self.checkpoint_per_epochs = checkpoint_per_epochs

    def on_epoch_end(self, epoch, logs):
        if epoch % self.checkpoint_per_epochs == 0:
            self.model.save_weights('{mp}/{mn}.{ts}.h5'.format(
                mp=self.save_path, mn=self.model_name, ts=int(time.time()))
            )


def get_class_weights(train_data, label_data):
    from sklearn.utils import compute_class_weight
    return compute_class_weight(class_weight='balanced', classes=train_data, y=label_data)


def train(data, labels, epochs, predict_steps, batch_size, num_labels, class_weight=None, data_dim=6):
    model = keras.models.Sequential()
    model.add(
        keras.layers.recurrent.LSTM(
            64, input_shape=(predict_steps, data_dim),
            return_sequences=True,
            # dropout=0.5, recurrent_dropout=0.5,
            activity_regularizer=keras.regularizers.l2(1.0)
        )
    )
    model.add(
        keras.layers.recurrent.LSTM(
            32, return_sequences=True,
            dropout=0.1, recurrent_dropout=0.1,
            activity_regularizer=keras.regularizers.l2(1.0)
        )
    )
    model.add(
        keras.layers.recurrent.LSTM(
            16,
            dropout=0.1, recurrent_dropout=0.1,
            activity_regularizer=keras.regularizers.l2(1.0)
        )
    )
    # model.add(keras.layers.core.Dense(64, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.core.Dense(32, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.core.Dense(num_labels, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    load_model_path = find_latest_model(MODEL_PATH, MODEL_NAME)
    if load_model_path:
        logger.info('load latest model: {p}'.format(p=load_model_path))
        model.load_weights(load_model_path)
    else:
        logger.info('train model from scratch')
    if class_weight:
        assert(len(class_weight.keys()) == num_labels)
    checkpointer = EpochCheckPoint(
        path=MODEL_PATH, model_name=MODEL_NAME, checkpoint_per_epochs=100)
    model.fit(
        data, labels, epochs=epochs,
        batch_size=batch_size, shuffle=True, validation_split=0.2,
        class_weight=class_weight, callbacks=[checkpointer]
    )
    return model


def gen_label(dataset, records):
    up_prec = 0.01
    predict_array = np.array([r[dataset.LAST_IDX] for r in records[:dataset.predict_steps]])
    price_mean = np.mean(predict_array)
    price_std = np.std(predict_array)
    up_bound = price_mean + price_std
    low_bound = price_mean - price_std

    label_last_prices = [r[dataset.LAST_IDX] for r in records[-dataset.label_steps:]]
    label_price_max = max(label_last_prices)
    label_price_min = min(label_last_prices)
    label = 0  # 价格平稳阶段
    if label_price_max < up_bound and label_price_min > low_bound:  # 波动收窄
        pass
    elif label_price_max > up_bound and label_price_min < low_bound:  # 波动扩大
        pass
    else:
        if (label_price_max - up_bound) / price_mean > up_prec:  # 价格突破顶部
            label = 1
        if (label_price_min - low_bound) / price_mean < -up_prec:  # 价格突破底部
            label = 2
    return label


def main(epochs, predict_steps, label_steps, size, batch_size=1, num_labels=3):
    dataset = MarketTickerDataSet()
    train_data, train_labels, test_data, test_labels = dataset.load_data(
        model_name=MODEL_NAME, gen_label_cb=gen_label, market_types=[1, 3], size=size,
        predict_steps=predict_steps, label_steps=label_steps, test_split_rate=0.1,
        num_labels=num_labels
    )
    labels = [i for one_hot in train_labels for i,e in enumerate(one_hot) if e == 1]
    class_weight = get_class_weights(np.unique(labels), labels)
    class_weight = dict([(i, w) for i, w in enumerate(class_weight)])
    logger.info('train class weight: {cw}'.format(cw=class_weight))
    model = train(
        data=train_data,
        labels=train_labels,
        epochs=epochs,
        predict_steps=predict_steps,
        batch_size=batch_size,
        num_labels=num_labels,
        class_weight=class_weight,
    )
    score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    logger.info('final score:{s}'.format(s=score))
    model.save_weights('{mp}/{mn}.{ts}.h5'.format(
        mp=MODEL_PATH, mn=MODEL_NAME, ts=int(time.time()))
    )

main(epochs=1000, size=6000, predict_steps=30, label_steps=20, batch_size=20)
