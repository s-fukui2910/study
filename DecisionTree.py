#coding: utf-8

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def main():
    dataset = datasets.load_iris()

    features = dataset.data
    targets = dataset.target

    for f,t in zip(features,targets):
        print(f,t)
        

    # Petal length と Petal width だけを特徴量として使う (二次元で図示したいので)
    petal_features = features[:, 2:]

    # 決定木の最大深度は制限しない
    clf = DecisionTreeClassifier()
    clf.fit(petal_features, targets)

    # 教師データの取りうる範囲 +-1 を計算する
    train_x_min = petal_features[:, 0].min() - 1
    train_y_min = petal_features[:, 1].min() - 1
    train_x_max = petal_features[:, 0].max() + 1
    train_y_max = petal_features[:, 1].max() + 1

    # 教師データの取りうる範囲でメッシュ状の座標を作る
    grid_interval = 0.2
    xx, yy = np.meshgrid(
        np.arange(train_x_min, train_x_max, grid_interval),
        np.arange(train_y_min, train_y_max, grid_interval),
    )

    # メッシュの座標を学習したモデルで判定させる
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 各点の判定結果をグラフに描画する
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.bone)

    # 教師データもプロットしておく
    for c in np.unique(targets):
        plt.scatter(petal_features[targets == c, 0],
                    petal_features[targets == c, 1])
    feature_names = dataset.feature_names
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    ##
    #plt.plot(range(10))
    cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    ##
    plt.show()

if __name__ == '__main__':
    main()
