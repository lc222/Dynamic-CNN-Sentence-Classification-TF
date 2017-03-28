import tensorflow as tf

class DCNN():
    def __init__(self, batch_size, sentence_length, num_filters, embed_size, top_k, k1):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.top_k = top_k
        self.k1 = k1

    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        self.k1 = k
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv_k_max_pooling"):
            for i in range(self.embed_size):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])
                #conv:[batch_size, sent_length+ws-1, num_filters]
                conv = tf.reshape(conv, [self.batch_size, self.num_filters[0], self.sentence_length])#[batch_size, sentence_length, num_filters]
                values = tf.nn.top_k(conv, k, sorted=False).values
                values = tf.reshape(values, [self.batch_size, k, self.num_filters[0]])
                #k_max pooling in axis=1
                convs.append(values)
            conv = tf.stack(convs, axis=2)
        #[batch_size, k1, embed_size, num_filters[0]]
        #print conv.get_shape()
        return conv

    def per_dim_conv_layer(self, x, w, b):
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        with tf.name_scope("per_dim_conv"):
            for i in range(len(input_unstack)):
                conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="SAME") + b_unstack[i])#[batch_size, k1+ws2-1, num_filters[1]]
                convs.append(conv)
            conv = tf.stack(convs, axis=2)
            #[batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    def fold_k_max_pooling(self, x, k):
        input_unstack = tf.unstack(x, axis=2)
        out = []
        with tf.name_scope("fold_k_max_pooling"):
            for i in range(0, len(input_unstack), 2):
                fold = tf.add(input_unstack[i], input_unstack[i+1])#[batch_size, k1, num_filters[1]]
                conv = tf.transpose(fold, perm=[0, 2, 1])
                values = tf.nn.top_k(conv, k, sorted=False).values #[batch_size, num_filters[1], top_k]
                values = tf.transpose(values, perm=[0, 2, 1])
                out.append(values)
            fold = tf.stack(out, axis=2)#[batch_size, k2, embed_size/2, num_filters[1]]
        return fold

    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(x, w) + b)
            h = tf.nn.dropout(h, dropout_keep_prob)
            o = tf.matmul(h, wo)
        return o

    def DCNN(self, sent, W1, W2, b1, b2, k1, top_k, Wh, bh, Wo, dropout_keep_prob):
        conv1 = self.per_dim_conv_layer(sent, W1, b1)
        conv1 = self.fold_k_max_pooling(conv1, k1)
        conv2 = self.per_dim_conv_layer(conv1, W2, b2)
        fold = self.fold_k_max_pooling(conv2, top_k)
        fold_flatten = tf.reshape(fold, [-1, top_k*100*14/4])
        print fold_flatten.get_shape()
        out = self.full_connect_layer(fold_flatten, Wh, bh, Wo, dropout_keep_prob)
        return out
