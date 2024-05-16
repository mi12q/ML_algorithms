import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, n_neighbors=3):
        """

        :param n_neighbors: - число соседей
        """
        self.X_train = None
        self.y_train = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Заролняет поля X_train и y_train.
        :param X: - входные данные
        :param y: - входные данные
        :return:
        """
        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        """
        Для каждой точки из тестовой выборки вызывает функцию 'predict_for_point'.
        :param X: - Набор данных
        :return:
        """

        return np.array([self.predict_for_point(x) for x in X])

    def predict_for_point(self, x):
        """
        Вычисляет расстояния между x и всеми точками в обучающем наборе,
        сортирует по расстоянию и находит индекси первых k соседей.
        Извлекает метки k ближайших соседей и возращает
        наиболее распространенную метку класса.
        :param x: - точка относительно которой вычисляем
        :return:
        """

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indexes = np.argsort(distances)[:self.n_neighbors]
        labels = self.y_train[k_indexes]
        labels_count = np.bincount(labels)

        return np.argmax(labels_count)


def main():
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    model = KNN(5)
    model.fit(X, y)
    y_prediction = model.predict(X)
    accuracy = accuracy_score(y, y_prediction)
    precision = precision_score(y,y_prediction)
    recall = recall_score(y, y_prediction)
    f1 = f1_score(y, y_prediction)
    print(f'Accuracy: {accuracy * 100}%')
    print(f'Precision: {precision* 100}%')
    print(f'Recall: {recall * 100}%')
    print(f'f1 score: {f1 * 100}%')
    h = .02
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    # Границы построены между классами
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_points, y_ponts = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
    predicted_data = model.predict(np.c_[x_points.ravel(), y_ponts.ravel()])
    predicted_data = predicted_data.reshape(x_points.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(x_points, y_ponts, predicted_data, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(x_points.min(), x_points.max())
    plt.ylim(y_ponts.min(), y_ponts.max())
    plt.title("Классификация KNN")
    plt.savefig('KNN.png')
    plt.show()


if __name__ == '__main__':
    main()
