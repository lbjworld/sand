# coding: utf-8
import logging
import keras
import time

from utils import find_latest_model
from dataset import MarketTickerDataSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = './models'
MODEL_NAME = 'p2_profit_detector'


def train(data, labels, epochs, predict_steps, batch_size, num_labels, data_dim=6):
    model = keras.models.Sequential()
    model.add(
        keras.layers.recurrent.LSTM(
            64, input_shape=(predict_steps, data_dim),
            return_sequences=True,
            # dropout=0.5, recurrent_dropout=0.5,
            # activity_regularizer=keras.regularizers.l1(0.2)
        )
    )
    model.add(
        keras.layers.recurrent.LSTM(
            32, return_sequences=True,
            # dropout=0.5, recurrent_dropout=0.5,
            activity_regularizer=keras.regularizers.l2(0.2)
        )
    )
    model.add(
        keras.layers.recurrent.LSTM(
            16,
            # dropout=0.5, recurrent_dropout=0.5,
            activity_regularizer=keras.regularizers.l2(0.2)
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
    model.fit(
        data, labels, epochs=epochs,
        batch_size=batch_size, shuffle=True, validation_split=0.2
    )
    return model


def gen_label(dataset, records):
    up_prec = 0.02
    label_buy_max = dataset.price_max(dataset.SELL_IDX, records[-dataset.label_steps:])
    last_price = records[:dataset.predict_steps][-1][dataset.LAST_IDX]
    label = 0
    if (label_buy_max - last_price) / last_price > up_prec:
        # 涨幅超过2%
        label = 1
    return label


def main(epochs, predict_steps, label_steps, size, batch_size=1):
    dataset = MarketTickerDataSet()
    train_data, train_labels, test_data, test_labels = dataset.load_data(
        model_name=MODEL_NAME, gen_label_cb=gen_label, market_types=[1, 2, 3], size=size,
        predict_steps=predict_steps, label_steps=label_steps, test_split_rate=0.1, num_labels=2
    )
    model = train(
        data=train_data,
        labels=train_labels,
        epochs=epochs,
        predict_steps=predict_steps,
        batch_size=batch_size,
        num_labels=2,
    )
    score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    logger.info('final score:{s}'.format(s=score))
    model.save_weights('{mp}/{mn}.{ts}.h5'.format(
        mp=MODEL_PATH, mn=MODEL_NAME, ts=int(time.time()))
    )

main(epochs=100, size=6000, predict_steps=20, label_steps=10, batch_size=20)