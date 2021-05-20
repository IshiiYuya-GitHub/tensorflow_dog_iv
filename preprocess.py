import tensorflow as tf

def resize(train, val, test, size):
    """サイズの変更

    Args:
        train (tfds): トレーニング
        val (tfds): 検証
        test (tfds): テスト
        size (tuple): サイズ 

    Returns:
        tfds: サイズ変更後のtfds
    """
    train = train.map(lambda image, label: (tf.image.resize(image, size), label))
    val = val.map(lambda image, label: (tf.image.resize(image, size), label))
    test = test.map(lambda image, label: (tf.image.resize(image, size), label))
    return (train, val, test)


def input_preprocess(image, label, NUM_CLASSES=120):
    """ワンホットラベルの設定

    Args:
        image (tf.Tensor): 画像
        label (tf.Tensor): ラベル
        NUM_CLASSES (int): 種類数

    Returns:
        tf.Tensor: 画像
        tf.Tensor: ワンホットラベル
    """
    
    label = tf.one_hot(label, NUM_CLASSES)
    return (image, label)


def batch_create(train, val, test, NUM_CLASSES=0, BATCH_SIZE=1, BUFFER_SIZE=1):
    """バッチ作成

    Args:
        train (tfds): トレーニング
        val (tfds): 検証
        test (tfds): テスト
        NUM_CLASSES (int): 種類数
        BATCH_SIZE (int): バッチ数
        BUFFER_SIZE(int): テストの枚数

    Returns:
        tfds: チューニング後のtfds
    """
    train = train.map(input_preprocess, num_parallel_calls=(tf.data.experimental.AUTOTUNE))
    train = train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    val = val.map(input_preprocess, num_parallel_calls=(tf.data.experimental.AUTOTUNE))
    val = val.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    test = test.map(input_preprocess, num_parallel_calls=(tf.data.experimental.AUTOTUNE))
    test = test.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE, drop_remainder=True)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return (train, val, test)

def shuffle(test, train=None, val=None, SHUFFLE_BUFFER_SIZE=1):
    """シャッフル

    Args:
        test (tfds): テスト
        train (tfds, optional): トレーニング. Defaults to None.
        val (tfds, optional): 検証. Defaults to None.
        SHUFFLE_BUFFER_SIZE (int, optional): シャッフル幅. Defaults to 1.

    Returns:
        tfds: シャッフル後のtfds
    """
    i=0
    if not train == None:
        train = train.shuffle(SHUFFLE_BUFFER_SIZE)
        i+=1
    if not val == None:
        val = val.shuffle(SHUFFLE_BUFFER_SIZE)
        i+=2
    test = test.shuffle(SHUFFLE_BUFFER_SIZE)
    if i == 0:
        return test
    if i == 1:
        return test, train
    if i == 2:
        return test, val
    if i == 3:
        return test, train, val

