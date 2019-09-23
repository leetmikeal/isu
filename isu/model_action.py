# -*- coding: utf-8 -*-
import os
import sys
import math

from tqdm import tqdm
import numpy as np


def load_numpy(numpy_cache_dir, use_for):
    import glob

    dir_numpy_cache_use_for = os.path.join(numpy_cache_dir, use_for, '*')
    numpy_cache_use_for_list = glob.glob(dir_numpy_cache_use_for)

    return numpy_cache_use_for_list


def iter_train_batch(image_list, label_list, batch_size):
    total = len(image_list)
    for start in range(0, total, batch_size):
        image_batch = image_list[start:min(start + batch_size, total)]
        label_batch = label_list[start:min(start + batch_size, total)]
        yield image_batch, label_batch

def iter_batch(image_list, batch_size):
    total = len(image_list)
    for start in range(0, total, batch_size):
        image_batch = image_list[start:min(start + batch_size, total)]
        yield image_batch


def train(
        model,
        image_training,
        image_validation,
        label_training,
        label_validation,
        epochs,
        batch_size,
        class_weight=None,
        callbacks=None,
        save_weight=False,
        seed=None):
    # # メモリ確保の方法を変更
    from keras import callbacks as cbks
    # from keras import backend as K
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    # コールバックの準備
    callback_metrics = None  # 内容が不明なので空とする
    callbacks = cbks.CallbackList(callbacks or [])
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        # 'steps': steps_per_epoch,
        # 'samples': num_train_samples,
        # 'verbose': verbose,
        'do_validation': True,
        'metrics': callback_metrics or [],
    })
    # コールバック実行
    callback_model.stop_training = False
    callbacks.on_train_begin()

    # import pandas as pd

    for i in range(epochs):

        callback_logs = {}
        callbacks.on_epoch_begin(epoch=i, logs=callback_logs)
        # print_epoch_start_time_training_validation(i + 1, epochs)

        train_batch_returns = []

        p = np.random.permutation(image_training.shape[0])
        image_training_shuffle = image_training[p]
        label_training_shuffle = label_training[p]

        # load numpy cache and train on batch
        train_generator = tqdm(
            iterable=iter_train_batch(image_training_shuffle, label_training_shuffle, batch_size),
            total=math.ceil(image_training_shuffle.shape[0] / batch_size))
        for j, [image_batch, label_batch] in enumerate(train_generator):
            # TODO: 2nd arguments 'batch_logs' must be implemented
            callbacks.on_batch_begin(j, {})

            # main process
            batch_return = model.train_on_batch(
                x=image_batch,
                y=label_batch,
                class_weight=class_weight)

            # result
            num_image = image_batch.shape[0]
            train_batch_returns.append(batch_return)
            # loss_sum += loss * num_image

            # TODO: 2nd arguments 'batch_logs' must be implemented
            callbacks.on_batch_end(j, {})

            # # 進捗を出力
            # # 累積のloss, accを出力
            # print_batch_remain_time_training_validation(time_start_batch=time_start_batch,
            #                                             current_batch=j + 1,
            #                                             batches=len(cache_image_training),
            #                                             loss_sum=loss_sum,
            #                                             acc_sum=acc_sum,
            #                                             num_image_total=num_image_total,
            #                                             )
        num_train_image_total = image_training_shuffle.shape[0]
        # train_loss = loss_sum / num_image_total

        # Validation after batch loop
        validate_batch_returns = []
        validate_generator = tqdm(
            iterable=iter_train_batch(image_validation, label_validation, batch_size), 
            total=math.ceil(image_validation.shape[0] / batch_size))
        for j, [image_batch, label_batch] in enumerate(validate_generator):

            # main process
            batch_return = model.test_on_batch(x=image_batch, y=label_batch)

            # result
            num_image = image_batch.shape[0]
            # loss_sum += loss * num_image
            validate_batch_returns.append(batch_return)

        num_val_image_total = image_validation.shape[0]
        # val_loss = loss_sum / num_image_total
        # print_batch_end_time_training_validation(val_loss=val_loss, val_acc=val_acc)
        # all batches end
        # print_all_batch_end_time_training_validation(time_start_batch=time_start_batch)

        import json
        from utility.save import check_dir
        weight_dump = model.get_weights()
        for widx, weight in enumerate(weight_dump):
            model_path = 'work/output/{:04d}/weight_{:02d}.json'.format(i, widx)
            check_dir(model_path)
            with open(model_path, 'w') as f:
                f.write(json.dumps(weight.tolist()))

        epoch_logs = {
            'batch_train_history': train_batch_returns,
            'batch_val_history': validate_batch_returns,
        }
        callbacks.on_epoch_end(i, epoch_logs)

        if callback_model.stop_training:
            break

    # # all epochs end
    # print_all_epoch_end_time_training_validation(epochs=epochs,
    #                                              time_start_epoch=time_start_epoch)

    callbacks.on_train_end()
    print('training finished')

    # # save weights
    # model.save_weights(weight_trained_path)



def validate(model, batch_size, verbose, source_image, result_path):
    import pandas as pd

    # 画像リストの読み込み
    df_definition_validate = source_image.extract('validation')
    if len(df_definition_validate) == 0:
        print('validation image is not found. {}'.format(source_image.path))
        sys.exit(1)

    # 画像の読み込み
    from utility.image import iter_cache_batch, load_batch
    predict_results = []
    size = model.input_shape[1:3]
    # channel = model.input_shape[3]
    classes = source_image.classes

    for i, rows in enumerate(
            iter_cache_batch(
            df_definition_validate, batch_size)):

        np_image, np_label = load_batch(rows, classes, size)
        # 実際の予測結果
        pred = model.predict(np_image, verbose=verbose, steps=1)
        predict_results.extend(pred)

    df_pred = pd.DataFrame(predict_results)
    # row_name_class_number = df_pred.columns.values

    # top-1のラベルの番号(0 start)を取得する
    df_pred.columns = list(range(len(classes)))  # 一時的に配列を使用する(0 start)
    series_validation = df_pred.idxmax(axis=1)  # series_validation
    series_validation.name = 'validation'
    df_pred.columns = classes  # csvのヘッダーには元のラベル名を使用する

    # 複数pathは1行にして保持する
    df_definition_validate_path = df_definition_validate.loc[:, 'path'].apply(
        lambda l: '|'.join(l))

    # 正解のラベルの番号(0 start)を取得する
    df_definition_validate_class = df_definition_validate.loc[:, classes]
    df_definition_validate_class.columns = list(range(len(classes)))
    series_answer = df_definition_validate_class.idxmax(
        axis=1)  # series_answer
    series_answer.name = 'answer'

    df_result = pd.concat([df_definition_validate_path,
                           df_pred, series_validation, series_answer], axis=1)
    df_result.to_csv(result_path, index=False, encoding='utf-8')

    print('validation finished')


def predict(
        model,
        image_unlabeled,
        result_dir_path='',
        batch_size=32,
        verbose=False):
    """predict
    
    Args:
        model (keras model): keras model
        image_unlabeled (numpy.array): 4 dimension numpy array [image_count, width, height, channel]
        result_dir_path (string, optional): saving result path (*.csv)
        batch_size (int, optional): predict batchsize
        verbose (bool, optional): output debug information. Defaults to False.
    
    Returns:
        [type]: [description]
    """

    import pandas as pd
    import numpy as np

    # cache
    # if os.path.exists(result_path):
    #     df_pred = pd.read_csv(result_path)
    #     return df_pred

    result_path = os.path.join(result_dir_path, 'predict.npy')
    if os.path.exists(result_path):
        pred = np.load(result_path)
        print('cached predicted : {}'.format(result_path))
        return pred

    if verbose:
        print('predicting... : {}'.format(image_unlabeled.shape))

    predict_generator = tqdm(
        iterable=iter_batch(image_unlabeled, batch_size),
        total=math.ceil(image_unlabeled.shape[0] / batch_size),
        disable=(not verbose))

    pred_list = []
    for j, image_batch in enumerate(predict_generator):
        pred = model.predict_on_batch(image_batch)
        pred_list.append(pred)

    # 実際の予測結果
    # pred = model.predict(
    #     image_unlabeled,
    #     batch_size=batch_size,
    #     verbose=verbose)
    # df_pred = pd.DataFrame(np.concatenate(pred_list))
    pred = np.array(pred_list)
    np.save(result_path, pred)

    # if result_path  != '' and os.path.exists(os.path.dirname(result_path)):
    #     df_pred.to_csv(result_path, index=False, encoding='utf-8')
    #     if verbose:
    #         print('saved result as csv : {}'.format(result_path))

    return pred
