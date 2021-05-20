import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import preprocess
import post_process
import matplotlib.pyplot as plt


dataset_name = "stanford_dogs"
(ds_train, ds_val, ds_test), ds_info = tfds.load(dataset_name,
                                                 split=["train", "test[:50%]", "test[50%:100%]"],
                                                 with_info=True,
                                                 as_supervised=True)

# 犬の種類数
NUM_CLASSES = ds_info.features["label"].num_classes
names = ds_info.features["label"].names
IMG_SIZE = 300
BATCH_SIZE = 100
size = (IMG_SIZE, IMG_SIZE)

ds_train, ds_val, ds_test = preprocess.resize(train=ds_train,
                                              val=ds_val,
                                              test=ds_test,
                                              size=size)


ds_train, ds_val, ds_test = preprocess.batch_create(train=ds_train, 
                                                            val=ds_val, 
                                                            test=ds_test,
                                                            NUM_CLASSES=NUM_CLASSES,
                                                            BATCH_SIZE=BATCH_SIZE,
                                                            BUFFER_SIZE=len(ds_test))

model, lr = post_process.build_model(NUM_CLASSES=NUM_CLASSES, IMG_SIZE=IMG_SIZE)

model.load_weights(post_process.checkpoint(num=0, IMG_SIZE=IMG_SIZE))

# es_callback = post_process.early_stopping()
# tb_callback = post_process.tensorboard(BATCH_SIZE=BATCH_SIZE, 
#                                        lr=lr, 
#                                        IMG_SIZE=IMG_SIZE)
# cp_callback = post_process.checkpoint_callback(num=0, IMG_SIZE=IMG_SIZE)

# epochs = 5000

# history = model.fit(ds_train,
#                     epochs=epochs,
#                     validation_data=ds_val,
#                     verbose=1,
#                     callbacks=[tb_callback, cp_callback, es_callback])

# # hitory = model.fit(ds_train,epochs=1, validation_data=ds_val, verbose=2)

# model = post_process.unfreeze_model(model)

# fine_epochs = 1000

# history = model.fit(ds_train,
#                     initial_epoch=epochs,
#                     epochs=epochs+fine_epochs,
#                     validation_data=ds_val,
#                     verbose=1,
#                     callbacks=[tb_callback, cp_callback, es_callback])

# loss, accuracy = model.evaluate(ds_test)
# print('test loss :', loss, 'Test accuracy :', accuracy)


# plt.figure(figsize=(10, 10))
# for image, label in ds_test.take(1):
#     plt.imshow(image[0])
#     plt.show()
    # prediction = model.predict(image)
    # print(np.argmax(prediction[0]))
    # print(tf.argmax(label[0]))

# 予測
label_names = []
for name in names:
  label_names.append(post_process.split_name(name))

image_batch, label_batch = ds_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
count = 0

plt.figure(figsize=(100, 100))
for i in range(BATCH_SIZE):
  print(np.argmax(predictions[i]))
  print(tf.argmax(label_batch[i]))
  if np.argmax(predictions[i]) == tf.argmax(label_batch[i]):
    count += 1
  ax = plt.subplot(10, 10, i+1)
  plt.imshow(image_batch[i].astype("uint8"))
  if np.argmax(predictions[i]) == tf.argmax(label_batch[i]):
    plt.title(label_names[tf.argmax((label_batch[i]))], fontsize=50, color='blue')
  else:
    plt.title('correct:' + label_names[tf.argmax((label_batch[i]))] + ', prediction:' + label_names[np.argmax(predictions[i])], fontsize=20, color='red') 
  plt.axis("off")
print(f'correct:{count}, uncorrect:{BATCH_SIZE-count}')
