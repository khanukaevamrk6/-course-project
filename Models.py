import numpy as np
from collections import Counter


class WDTW:

    def __init__(self, g=0.05):
        self.g = g

    def weight(self, diff, m):

        return 1.0 / (
            1.0 + np.exp(-self.g * (diff - m))
        )

    def distance(self, ts1, ts2):

        n = len(ts1)
        m = len(ts2)

        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        midpoint = max(n, m) / 2

        for i in range(1, n + 1):
            for j in range(1, m + 1):

                weight = self.weight(
                    abs(i - j),
                    midpoint
                )

                dist = weight * (
                    ts1[i - 1] - ts2[j - 1]
                ) ** 2

                cost[i, j] = dist + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1]
                )

        return np.sqrt(cost[n, m])

class WDTWClassifier:

    def __init__(self, k=1, g=0.05):

        self.k = k
        self.wdtw = WDTW(g)

    def fit(self, train_loader):

        X_all = []
        y_all = []

        for X_batch, y_batch in train_loader:
            X_all.append(
                X_batch.numpy()
            )

            y_all.append(
                y_batch.numpy()
            )

        self.X_train = np.concatenate(X_all)
        self.y_train = np.concatenate(y_all)

    def predict_one(self, sample):

        distances = []

        for idx, train_sample in enumerate(
                self.X_train
        ):
            # если shape=(seq_len,1)
            x1 = sample.squeeze()
            x2 = train_sample.squeeze()

            d = self.wdtw.distance(
                x1,
                x2
            )

            distances.append(
                (d, self.y_train[idx])
            )

        distances.sort(
            key=lambda x: x[0]
        )

        neighbors = distances[:self.k]

        labels = [
            label
            for _, label in neighbors
        ]

        return Counter(labels).most_common(1)[0][0]

    def predict(self, test_loader):

        predictions = []

        for X_batch, _ in test_loader:

            for sample in X_batch:
                pred = self.predict_one(
                    sample.numpy()
                )

                predictions.append(pred)

        return np.array(predictions)