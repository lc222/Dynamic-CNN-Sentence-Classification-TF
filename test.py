import tensorflow as tf
import numpy as np
import  embedding as emb
def qselect(A, k):
    if len(A) < k: return A
    pivot = A[-1]
    right = [pivot] + [x for x in A[:-1] if x >= pivot]
    rlen = len(right)
    if rlen == k:
        return right
    if rlen > k:
        return qselect(right, k)
    else:
        left = [x for x in A[:-1] if x < pivot]
        return qselect(left, k - rlen) + right

# a = np.array([[1,2,3], [3,4,5]])
# print a.shape
# a = tf.placeholder(tf.float32, [120])
# b = tf.reshape(a, [2,3,4,5])
# values, indices = tf.nn.top_k(b, 2)
# with tf.Session() as sess:
#     print sess.run(b, feed_dict={a:np.arange(120, dtype="float32")})
#     print sess.run(tf.nn.top_k(b, 2, sorted=False), feed_dict={a:np.arange(120, dtype="float32")})

embed_dim = 50
ws = [4, 5]
top_k = 4
k1 = 5
num_filters = [3, 14]
batch_size = 2
num_hidden = 100
sentence_length = 10
num_class = 6
lr = 0.01


def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

glove = emb.GloVe(N=embed_dim)

with tf.Session() as sess:
    sent = tf.placeholder(tf.int32, [batch_size, sentence_length])

    sent_embed = tf.nn.embedding_lookup(glove.g, sent)
    input_x = tf.reshape(sent_embed, [batch_size, sentence_length, embed_dim, 1])

    W1 = init_weights([ws[0], embed_dim, 1, num_filters[0]], "W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0], embed_dim]), "b1")
    init = tf.global_variables_initializer().run()
    print W1.eval(), b1.eval()

    input_unstack = tf.unstack(input_x, axis=2)
    w_unstack = tf.unstack(W1, axis=1)
    b_unstack = tf.unstack(b1, axis=1)
    convs = []

    conv = tf.nn.relu(tf.nn.conv1d(input_unstack[0], w_unstack[0], stride=1, padding="SAME") + b_unstack[0])
    #print conv.eval()
    # conv:[batch_size, sent_length+ws-1, num_filters]
    conv1 = tf.reshape(conv, [batch_size, num_filters[0],
                             sentence_length])  # [batch_size, sentence_length, num_filters]
    values, indices = tf.nn.top_k(conv1, k1, sorted=False)
    #print values.eval()
    values1 = tf.reshape(values, [batch_size, k1, num_filters[0]])
    # k_max pooling in axis=1
    convs.append(values1)
    conv2 = tf.stack(convs, axis=2)

    a, b, c ,d, e = sess.run([input_x, conv, conv1, values, indices], feed_dict={sent:[[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]})
    print a,b,c,d,e