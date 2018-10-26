#coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def sin_model(x, sigma=0.05):
    #大きな波＋小さな波＋ノイズからなるダミーデータ
    noise = sigma * np.random.randn(len(x))
    return np.sin(x) + noise
        
# Step 1. Prepare data
#x_data = np.linspace(0, 2 * np.pi, 100)[:, None]
#y_data = np.sin(x_data)
##add random noize
x = np.linspace(0, 4 * np.pi, 100)
#x = 10 * np.random.rand(100)
x_data = x[:, None]
y = sin_model(x)
y_data = y[:, None]

# Step 2. Define operation
x = tf.placeholder(tf.float32, (None, 1))
y_answer = tf.placeholder(tf.float32)

n_var = 100

w1 = tf.Variable(tf.truncated_normal([1, n_var], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[n_var]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([n_var, n_var], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[n_var]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.Variable(tf.truncated_normal([n_var, 1], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[1]))
y_model = tf.matmul(h2, w3) + b3

loss = tf.reduce_mean((y_model - y_answer)**2)
train = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()


# Step 3. Run operation
session = tf.Session()
session.run(init)
for i in range(10000):
    session.run(train, {x: x_data, y_answer: y_data})
    if i % 1000 == 0:
        current_loss, current_y_model = session.run(
            [loss, y_model], {x: x_data, y_answer: y_data})
        print("Loss: ",current_loss)
current_loss, current_y_model = session.run(
    [loss, y_model], {x: x_data, y_answer: y_data})

plt.plot(x_data, y_data, '.', label='Answer')
plt.plot(x_data, current_y_model, '.-', label='Model')
plt.legend()
plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
plt.show()
