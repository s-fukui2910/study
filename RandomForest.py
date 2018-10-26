#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

# データの用意
iris = sklearn.datasets.load_iris()
features = iris.data[:, [0, 2]]
#features = iris.data[:, :]
targets = iris.target

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(features, iris.target, test_size=0.3)


# 学習
rf = sklearn.ensemble.RandomForestClassifier(100)
rf.fit(train_x, train_y)

# 学習
#rf = sklearn.ensemble.RandomForestClassifier(100)
#rf.fit(features, targets)

# 評価
accuracy = rf.score(test_x, test_y)
print('accuracy {0:.2%}'.format(accuracy))


# 結果のプロット
prediction = rf.predict(test_x)

# 教師データの取りうる範囲 +-1 を計算する
train_x_min = features[:, 0].min() - 1
train_y_min = features[:, 1].min() - 1
train_x_max = features[:, 0].max() + 1
train_y_max = features[:, 1].max() + 1

# 教師データの取りうる範囲でメッシュ状の座標を作る
grid_interval = 0.1
xx, yy = np.meshgrid(
    np.arange(train_x_min, train_x_max, grid_interval),
    np.arange(train_y_min, train_y_max, grid_interval),
)

# 教師データの取りうる範囲でメッシュ状の座標を作る
#grid_interval = 0.1
#xx, yy = np.meshgrid(
#    np.arange(0, 10, grid_interval),
#    np.arange(0, 10, grid_interval),
#)

# メッシュの座標を学習したモデルで判定させる
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
#print(Z)
# 各点の判定結果をグラフに描画する
plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.bone)

plt.scatter(*test_x.T, c=[['orange', 'green', 'blue'][answer] if answer == predict else 'red' for answer, predict in zip(test_y, prediction)])
plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
plt.show()
