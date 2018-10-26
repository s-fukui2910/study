#coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def sin_model(x, sigma=0.10):
    #大きな波＋小さな波＋ノイズからなるダミーデータ
    noise = sigma * np.random.randn(len(x))
    return np.sin(x) + noise

def neuralnet(X,Y):
    x_data = X[:, None]
    y_data = Y[:, None]
    
    x = tf.placeholder(tf.float32, (None, 1))
    y_answer = tf.placeholder(tf.float32)
    
    n_var = 100
    #入力(全結合)層 1→n_var
    w1 = tf.Variable(tf.truncated_normal([1, n_var], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[n_var]))
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    #全結合層 n_var→_n_var
    w2 = tf.Variable(tf.truncated_normal([n_var, n_var], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[n_var]))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    #出力層 n_var→1
    w3 = tf.Variable(tf.truncated_normal([n_var, 1], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[1]))
    y_model = tf.matmul(h2, w3) + b3
    
    loss = tf.reduce_mean((y_model - y_answer)**2)
    train = tf.train.AdamOptimizer().minimize(loss)
    
    init = tf.global_variables_initializer()

    
    session = tf.Session()
    session.run(init)
    for i in range(10001):
        session.run(train, {x: x_data, y_answer: y_data})
        if i % 1000 == 0:
            current_loss, current_y_model = session.run(
                [loss, y_model], {x: x_data, y_answer: y_data})
            print("Loss: ",current_loss)
    return x_data,current_y_model

def randomforest(x,y):
    # ランダムフォレスト実行
    rfr = RandomForestRegressor(100)  # インスタンスの生成　木の数を100個に指定
    rfr.fit(x[:, None], y)            # 学習実行
    # 確認用に0〜10の1000個のデータを用意
    xfit = np.linspace(0, 4 * np.pi, 1000)       #0〜10まで1000個
    yfit = rfr.predict(xfit[:, None]) # 予測実行
    return xfit,yfit


# 入力
x = np.linspace(0, 4 * np.pi, 100)
y = sin_model(x)
#NNによる予測
nx,ny = neuralnet(x,y)
#ランダムフォレストによる予測
rx,ry =randomforest(x,y)
#ランダムフォレストの入力にNNの出力を利用して予測
tmpx=np.resize(nx,(1,-1))
tmpy=np.resize(ny,(1,-1))
nrx,nry =randomforest(tmpx[0],tmpy[0])
# 結果比較用に実際の値を取得(ノイズ無し)
xtrue = np.linspace(0, 4 * np.pi, 1000)       #0〜10まで1000個
ytrue = sin_model(xtrue,0) # xfitを波発生関数に食わせて、その結果を取得
# 結果確認
plt.figure(figsize = (16,8))
plt.plot(x, y, '.k', label='data')
plt.plot(xtrue,ytrue, '-y', label='answer')
plt.plot(nx, ny, '-b', label='nn')
plt.plot(rx, ry, '-g', label='rf')
plt.plot(nrx,nry, '-r', label='nrf')
plt.legend()
plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
plt.show()
