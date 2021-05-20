from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime


def build_model(NUM_CLASSES, IMG_SIZE):
    """モデルの構築

    Args:
        NUM_CLASSES (int): 種類数
        IMG_SIZE (int): サイズ

    Returns:
        tf.keras.Model: モデル
        int: Adamのハイパーパラメータ
    """
    img_augmentation = Sequential([preprocessing.RandomRotation(factor=0.15),
                                   preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                                   preprocessing.RandomFlip(),
                                   preprocessing.RandomContrast(factor=0.1)],
                                   name='img_augmentation')                                   
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB3(include_top=False,
                           input_tensor=x, 
                           weights='imagenet')
    model.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool')(model.output)
    x = BatchNormalization(trainable=True)(x)
    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='pred')(x)
    model = Model(inputs, outputs, name='EfficientNet')
    lr = 1e-4
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return (model, lr)


def unfreeze_model(model):
    """fine tuning

    Args:
        model (tf.keras.Model)): 転移学習済みのモデル

    Returns:
        tf.keras.Model: fine tuningの設定済みモデル
    """
    for layer in model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def early_stopping():
    """過学習防止

    Returns:
        tf.keras.callbacks.Early_stopping: 改善が見られなければ学習を止める
    """
    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    return es_callback


def tensorboard(BATCH_SIZE, lr, IMG_SIZE):
    """学習の記録保管

    Args:
        BATCH_SIZE (int): バッチ数
        lr (int): 学習係数

    Returns:
        tf.keras.callbacks.Tensorborad: 記録
    """
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + \
              'img_size' + str(IMG_SIZE) + 'batch-' + \
               str(BATCH_SIZE) + 'learningrate-' + str(lr)
    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tb_callback


def checkpoint_callback(num, IMG_SIZE):
    """モデルの保存

    Args:
        num (int): ナンバリング

    Returns:
        tf.keras.callbacks.ModelCheckpoint: モデルの保存
    """
    checkpoint_path = f"training_{num}_{IMG_SIZE}/cp.ckpt"
    cp_callback = ModelCheckpoint(checkpoint_path,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  save_weights_only=True)
    return cp_callback


def checkpoint(num, IMG_SIZE):
    """チェックポイントからのロード

    Args:
        num (int): ナンバリング
        IMG_SIZE (int): サイズ

    Returns:
        str: ファイル名
    """
    return f"training_{num}_{IMG_SIZE}/cp.ckpt"


def split_name(name):
    """ラベルの可読性をあげる

    Args:
        name (str): ラベルの名前

    Returns:
        str: 一般的な名前
    """
    return name.split('-')[1]