import tensorflow as tf
# tf.enable_eager_execution()
import tensorflow.contrib.layers as layers
import numpy as np
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt

DATA_FILE_TEMPLATE = './data/{}_data.mat'
LABEL_FILE_TEMPLATE = './data/{}_label.mat'
data_file = DATA_FILE_TEMPLATE.format('train')
label_file = LABEL_FILE_TEMPLATE.format('train')

data = scio.loadmat('./data/train_data.mat')['train_data'].astype(np.float32)
label = scio.loadmat('./data/train_label.mat')['train_label'].astype(np.int64)

label = np.reshape(label, (-1))
for i in range(label.shape[0]):
    if label[i] == -1:
        label[i] = 0

batch_size =128
x = tf.placeholder(shape=[None, 361], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)
is_training = tf.placeholder(shape=(), dtype=tf.bool)

# istraining = tf.placeholder(shape=[],dtype=tf.bool)
def net(input):
    y = layers.fully_connected(input, 256,
                               # activation_fn=tf.nn.relu,
                               weights_initializer=tf.truncated_normal_initializer(),
                               weights_regularizer=layers.l2_regularizer(scale=0.0001))
    # y = layers.batch_norm(y, is_training=is_training, activation_fn=tf.nn.relu)
    # y = layers.batch_norm(y, is_training=is_training, activation_fn=tf.nn.relu)
    # y = layers.fully_connected(y, 512,
    #                            # activation_fn=tf.nn.relu,
    #                            weights_initializer=tf.truncated_normal_initializer(),
    #                            weights_regularizer=layers.l2_regularizer(scale=0.0001))

    # y = layers.batch_norm(y, is_training=is_training, activation_fn=tf.nn.relu)
    y = layers.fully_connected(y, 512,
                               # activation_fn=tf.nn.relu,
                               weights_initializer=tf.truncated_normal_initializer(),
                               weights_regularizer=layers.l2_regularizer(scale=0.0001))

    # y = layers.fully_connected(y, 1024,
    #                            # activation_fn=tf.nn.relu,
    #                            weights_initializer=tf.truncated_normal_initializer(),
    #                            weights_regularizer=layers.l2_regularizer(scale=0.0001))

    # y = layers.batch_norm(y, is_training=is_training, activation_fn=tf.nn.relu)
    # y = layers.dropout(y, is_training=is_training)
    y = layers.dropout(y, keep_prob=0.1, is_training=is_training)
    y = layers.fully_connected(y, 2, activation_fn=None,
                               weights_initializer=tf.truncated_normal_initializer())

    return y

logits = net(x)

# tf.add_to_collection('network-output', logits)

# gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
# learning_rate = tf.train.exponential_decay(0.0001
#                                            , gloabl_steps,
#                                            1000,
#                                            0.00001,
#                                            staircase=True)

predictions = tf.argmax(logits, axis=-1)

acc, update = tf.metrics.accuracy(labels=y, predictions=tf.argmax(logits, axis=-1))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)#0.83809525
# opt = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)#0.84761906
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)#0.85714287
# opt = tf.train.AdamOptimizer(0.001)#0.85714287

# opt = tf.train.RMSPropOptimizer(0.00001, 0.9)
# opt = tf.train.GradientDescentOptimizer(learning_rate)
update_op = tf.get_default_graph().get_collection(name=tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_op):
    train_step = opt.minimize(loss)
saver = tf.train.Saver()
# train_step = opt.minimize(loss,global_step=gloabl_steps)


def get_batch(data, labels):
    id = np.random.randint(low=0, high=labels.shape[0], size=batch_size, dtype=np.int32)
    return data[id , ...],labels[id]


    # img = data
    # for i in range(img.shape[0]):
    #     im = img[i].reshape(19, 19)
    #     # im = ...
    #     # img[i] = im.reshape(-1)
    #     im = Image.fromarray(np.uint8(im * 255))
    #     pattern = [Image.FLIP_TOP_BOTTOM,Image.FLIP_LEFT_RIGHT,Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
    #     im = im.transpose(pattern[i%5])
    #     im = np.array(im)
    #     img[i] = im.reshape(-1)
    #
    #     # plt.imshow(im,cmap='gray')
    #     # plt.show()
    # img = img[id,...]
    # l = labels[id]
    # return img, l

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for i in range(2001):
        d, l = get_batch(data, label)

        if i % 100 == 0:
            print("step:", i, ',loss: %.18f'%sess.run(loss, feed_dict={x: d, y: l, is_training: False}))

            # learning_rate_val = sess.run(learning_rate)
            # print('learning_rate:',learning_rate_val)
            # saver.save(sess, "./save/model%d.ckpt"%i)
        sess.run(train_step, feed_dict={x: d, y: l, is_training: True})


    d = scio.loadmat('./data/test_data.mat')['test_data'].astype(np.float32)

    predict = sess.run(predictions, feed_dict={x: d, is_training: False})
    predict = predict.tolist()
    print('predict:', predict)
