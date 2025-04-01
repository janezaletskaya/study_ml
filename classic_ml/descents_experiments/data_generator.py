import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self, random_state=42):
        self.random_state = random_state

    def generate_linear_data(self, n_samples: int = 1000, n_features: int = 1, noise: float = 0.3) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует линейные данные с добавлением гауссовского шума.

        Parameters
        ----------
        n_samples : int
            Количество объектов.
        n_features : int
            Количество признаков.
        noise : float
            Уровень гауссовского шума.

        Returns
        -------
        X : np.ndarray
            Матрица признаков (n_samples x n_features).
        y : np.ndarray
            Целевые значения (n_samples,).
        true_weights : np.ndarray
            Истинные веса, использованные для генерации данных.
        """
        np.random.seed(self.random_state)
        true_weights = np.random.randn(n_features + 1)

        X = np.random.randn(n_samples, n_features)
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        y = X_with_bias.dot(true_weights) + noise * np.random.randn(n_samples)

        return X, y, true_weights

    def generate_outliers_data(self, n_samples=1000, n_features=1, outlier_ratio=0.05, outlier_scale=10, noise=0.3):
        """
        Генерирует данные с выбросами.

        Parameters:
        -----------
        n_samples : int
            Количество образцов
        n_features : int
            Количество признаков
        outlier_ratio : float
            Доля выбросов
        outlier_scale : float
            Масштаб выбросов
        noise : float
            Уровень шума

        Returns:
        --------
        X : numpy.ndarray
            Матрица признаков
        y : numpy.ndarray
            Целевые значения
        """
        np.random.seed(self.random_state)

        X, y, true_weights = self.generate_linear_data(n_samples, n_features, noise)

        n_outliers = int(outlier_ratio * n_samples)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        y[outlier_indices] += outlier_scale * np.random.randn(n_outliers)

        return X, y, true_weights

    def generate_ill_conditioned_data(self, n_samples=1000, n_features=10, condition_number=1000, noise=0.3):
        """
        Генерирует плохо обусловленные данные.

        Parameters:
        -----------
        n_samples : int
            Количество образцов
        n_features : int
            Количество признаков
        condition_number : float
            Число обусловленности матрицы признаков
        noise : float
            Уровень шума

        Returns:
        --------
        X : numpy.ndarray
            Матрица признаков
        y : numpy.ndarray
            Целевые значения
        """
        np.random.seed(self.random_state)

        X = np.random.randn(n_samples, n_features)

        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s[0] = condition_number * s[-1]
        s[1:-1] = np.geomspace(s[0], s[-1], n_features - 2)
        X = U.dot(np.diag(s).dot(Vt))

        true_weights = np.random.randn(n_features)
        y = X.dot(true_weights) + noise * np.random.randn(n_samples)

        return X, y, true_weights

    def generate_nonlinear_data(self, n_samples=1000, noise=0.3):
        """
        Генерирует сильно нелинейные данные с одним признаком.

        Parameters:
        -----------
        n_samples : int
            Количество образцов
        noise : float
            Уровень шума

        Returns:
        --------
        X : numpy.ndarray
            Матрица признаков (n_samples, 1)
        y : numpy.ndarray
            Целевые значения
        """
        np.random.seed(self.random_state)

        X = np.random.uniform(-5, 5, (n_samples, 1))
        x = X[:, 0]

        y = (
                np.sin(3 * x) +
                0.1 * x ** 3 - 0.5 * x +
                noise * np.random.randn(n_samples)
        )

        return X, y

    def generate_multimodal_data(self, n_samples=1000, n_features=1, n_modes=3, noise=0.3):
        """
        Генерирует мультимодальные данные (несколько кластеров).

        Parameters:
        -----------
        n_samples : int
            Количество образцов
        n_features : int
            Количество признаков
        n_modes : int
            Количество мод (кластеров)
        noise : float
            Уровень шума

        Returns:
        --------
        X : numpy.ndarray
            Матрица признаков
        y : numpy.ndarray
            Целевые значения
        """
        np.random.seed(self.random_state)

        samples_per_mode = n_samples // n_modes

        X_list = []
        y_list = []

        for i in range(n_modes):
            center_x = (i - n_modes // 2) * 5
            center_y = (i - n_modes // 2) * 3

            X_mode = np.random.randn(samples_per_mode, n_features) + center_x
            y_mode = center_y + np.sin(X_mode[:, 0]) + noise * np.random.randn(samples_per_mode)

            X_list.append(X_mode)
            y_list.append(y_mode)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    @staticmethod
    def visualize_data(X, y, title="Synthetic Data"):
        """
        Визуализирует сгенерированные данные.

        Parameters:
        -----------
        X : numpy.ndarray
            Матрица признаков
        y : numpy.ndarray
            Целевые значения
        title : str
            Заголовок графика
        save_path : str, optional
            Путь для сохранения изображения
        """
        plt.figure(figsize=(10, 6))

        if X.shape[1] == 1:
            plt.scatter(X[:, 0], y, alpha=0.6)
            plt.xlabel('X')
            plt.ylabel('y')
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(label='y')

        plt.title(title)
        plt.grid(True, alpha=0.3)

        plt.show()



