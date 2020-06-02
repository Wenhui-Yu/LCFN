import tensorflow as tf
import numpy as np

sess=tf.Session()

l = 3
e = 2

users = tf.Variable(tf.random_normal([l, e], mean=0.01, stddev=0.02, dtype=tf.float32))
#items = tf.Variable(tf.random_normal([l, e], mean=0.01, stddev=0.02, dtype=tf.float32))
ones = np.ones((1, l)).astype(np.float32)
d = tf.matmul(tf.matmul(ones, users), users, transpose_a=False, transpose_b=True)
d = tf.diag(tf.pow(d[0], -1))
a = tf.matmul(users, users, transpose_a=False, transpose_b=True)
aa = tf.matmul(d, a)


sess.run(tf.global_variables_initializer())
print(sess.run(users))
print(sess.run(a))
print(sess.run(d))
print(sess.run(aa))
print(sess.run(aa))
c = sess.run(aa)
print(c[0])
print(sum(c[0]))


'''
sess=tf.Session()

u = tf.placeholder(tf.int32, shape=(None,))
i = tf.placeholder(tf.int32, shape=(None,))
l = 3
e = 2
users = tf.Variable(tf.random_normal([l, e], mean=0.01, stddev=0.02, dtype=tf.float32))
items = tf.Variable(tf.random_normal([l, e], mean=0.01, stddev=0.02, dtype=tf.float32))
b = tf.pow(users, -1)
a = 1/users

u_GMF = tf.nn.embedding_lookup(users, u)
i_GMF = tf.nn.embedding_lookup(items, i)

if u_GMF.get_shape().as_list()[0] == None:length = 1
else:length = u_GMF.get_shape().as_list()[0]

ones = np.array([1] * length).astype(np.float32)
pos = tf.reduce_sum(tf.multiply(u_GMF, i_GMF), axis=1)
Loss = ones-pos
loss = tf.nn.l2_loss(Loss)

sess.run(tf.global_variables_initializer())
index = np.array([[1,0],[1,1],[1,2]])
print(sess.run(u , feed_dict={u: index[:,0],i: index[:,1]}))

print('u', sess.run(users , feed_dict={u: index[:,0],i: index[:,1]}))
print('a', sess.run(a , feed_dict={u: index[:,0],i: index[:,1]}))
print('b', sess.run(b , feed_dict={u: index[:,0],i: index[:,1]}))
print(sess.run(items , feed_dict={u: index[:,0],i: index[:,1]}))
print(sess.run(u_GMF , feed_dict={u: index[:,0],i: index[:,1]}))
print(sess.run(i_GMF , feed_dict={u: index[:,0],i: index[:,1]}))

print(sess.run(pos , feed_dict={u: index[:,0],i: index[:,1]}))
print(sess.run(Loss , feed_dict={u: index[:,0],i: index[:,1]}))
print(sess.run(loss , feed_dict={u: index[:,0],i: index[:,1]}))

'''