import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import graphviz


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """

        :param feature_index: - индекс признака, по которому разбивается вершина
        :param threshold: - пороговое значение, по которому разбивается вершина
        :param left: - левое поддерево
        :param right: - правое поддерево
        :param value: - значение в листовой вершине
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, features_count=None):
        """

        :param min_samples_split: - минимальное количество образцов, необходимых для разделения узла
        :param max_depth: - максимальная глубина дерева
        :param features_count: - количество признаков для рассмотрения при поиске лучшего разделения
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.features_count = features_count
        self.root = None

    def fit(self, X, y):
        """
        Обучает дерево на входных данных, выращивая дерево рекурсивно с помощью метода build_tree.
        :param X: - входные данные
        :param y: - входные данные
        :return:
        """
        self.features_count = X.shape[1] if self.features_count is None else min(self.features_count, X.shape[1])
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        """
        1) Проверяет, достигнута ли максимальная глубина, есть ли только одна метка в данных
        или количество образцов меньше минимального, необходимого для разделения (min_samples_split).
        Создается листовой узел с наиболее распространенной меткой данных, если какое то из условий выполнено.
        2) Случайным образом выбирает множество признаков,
         лучший признак и порог находятся с помощью метода _best_criteria.
        3) Данные разделяются с использованием лучшего признака и порога,
        рекурсивно вызывается метод _grow_tree для левого и правого подмножеств.
        Создается новый узел с лучшим признаком, порогом, левым и правым узлами.

        :param X: - входные данные
        :param y: - входные данные
        :param depth: - глубина
        :return:
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_index = np.random.choice(n_features, self.features_count, replace=False)

        best_feature, best_thresh = self.check_best_criteria(X, y, feature_index)

        left_index, right_index = self.split_data(X[:, best_feature], best_thresh)
        left = self.build_tree(X[left_index, :], y[left_index], depth + 1)
        right = self.build_tree(X[right_index, :], y[right_index], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def check_best_criteria(self, X, y, feature_index):
        """
        Ищет лучший признак и порог, перебирая выбранные признаки и пороги.
        :param X: - входные данные
        :param y: - входные данные
        :param feature_index: - индекс признака
        :return:
        """
        best_gain = -1
        split_index, split_thresh = None, None
        for feat_idx in feature_index:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.calculate_information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_idx
                    split_thresh = threshold
        return split_index, split_thresh

    def split_data(self, X_column, split_thresh):
        """
        Разделяет выборки по пороговому значению
        :param X_column:
        :param split_thresh:
        :return:
        """
        left_idexes = np.argwhere(X_column <= split_thresh).flatten()
        right_idexes = np.argwhere(X_column > split_thresh).flatten()
        return left_idexes, right_idexes

    def calculate_information_gain(self, y, X_column, split_thresh):
        """
        Информационный выигрыш вычисляется как разница между энтропией до и после разделения данных.
        :param y: - входные данные
        :param X_column: - - входные данные
        :param split_thresh: - пороговоe значениe
        :return:
        """
        parent_entropy = self.calculate_entropy(y)
        left_idxs, right_idxs = self.split_data(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self.calculate_entropy(y[left_idxs]), self.calculate_entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def calculate_entropy(self, y):
        """
        Вычисляет энтропию выборки
        :param y: - выборка
        :return:
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def _most_common_label(self, y):
        """
        Возвращает наиболее часто встречающееся значение в выборке
        :param y:
        :return:
        """
        _, counts = np.unique(y, return_counts=True)
        return max(zip(_, counts), key=lambda x: x[1])[0]

    def predict(self, X):
        """
        Прогнозирует метки для новых данных
        :param X: - новые данные
        :return:
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """

        :param x:
        :param node:
        :return:
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def visualize_tree(self):
        """
        Использует рекурсию для обхода узлов дерева и добавляет их в объект Digraph из библиотеки graphviz.
        Если узел содержит значение (лист), добавляем его значение в качестве меткиб
        иначе добавляем условие разбиения (порог и номер признака)
        Если у узла есть левый потомок, добавляем ребро и вызываем `add_nodes` для левого потомка,
        если у узла есть правый потомок, добавляем ребро и вызываем `add_nodes` для правого потомка.
        :return: - возвращает объект `Digraph`
        """
        dot = graphviz.Digraph()

        def add_nodes(node):
            if node.value is not None:
                label = str(node.value)
            else:
                label = "X[" + str(node.feature_index) + "] <= " + str(node.threshold)

            dot.node(str(id(node)), label)

            if node.left is not None:
                dot.edge(str(id(node)), str(id(node.left)))
                add_nodes(node.left)

            if node.right is not None:
                dot.edge(str(id(node)), str(id(node.right)))
                add_nodes(node.right)

        add_nodes(self.root)
        return dot


def main():

    X, y = make_classification(n_samples=1400, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    my_three = DecisionTree(max_depth=3)
    my_three.fit(X, y)

    y_prediction = my_three.predict(X)
    print(classification_report(y, y_prediction))

    my_three.visualize_tree()

    #Границы построены между классами
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_points, y_points = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    data = my_three.predict(np.c_[x_points.ravel(), y_points.ravel()])
    data = data.reshape(x_points.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x_points, y_points, data, cmap='bwr', alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.xlim(x_points.min(), x_points.max())
    plt.ylim(y_points.min(), y_points.max())
    plt.title("Классификация дерева решений")
    plt.savefig('decision_three.png')
    plt.show()


if __name__ == '__main__':
    main()
