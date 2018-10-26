#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def main():
    dataset = datasets.load_iris()

    features = dataset.data
    targets = dataset.target

    # Petal length と Petal width だけを特徴量として使う
    petal_features = features[:, 2:]

    # モデルを学習させる
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(petal_features, targets)

    # DOT 言語のフォーマットで決定木の形を出力する
    with open('iris-dtree.dot', mode='w') as f:
        tree.export_graphviz(clf, out_file=f)


if __name__ == '__main__':
    main()
