import math
import numpy as np


class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=3, buckets=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.buckets = buckets
        self.trees = []
        self.y_mean = 0

    def fit(self, X, y):
        self.y_mean = np.mean(y)
        y_pred = np.full_like(y, self.y_mean, dtype=np.float32)

        lead_nodes = 0 # 每颗树叶子结点数目
        lead_scores = 0 # 每颗树叶子结点分数

        for i in range(self.n_estimators):
            gradient = self.loss_gradient(y, y_pred)
            tree = DecisionTree(
                max_depth=self.max_depth,
                lead_nodes=lead_nodes,
                lead_scores=lead_scores,
                buckets=self.buckets
            )
            tree.fit(X, gradient)

            lead_nodes = tree.lead_nodes
            lead_scores = tree.lead_scores

            self.trees.append(tree)
            update = tree.predict(X)
            # shrinkage
            y_pred += self.learning_rate * update

    def predict(self, X):
        y_pred = np.full(len(X), self.y_mean)

        for tree in self.trees:
            update = tree.predict(X)
            y_pred += self.learning_rate * update

        return y_pred

    def loss_gradient(self, y_true, y_pred):
        return y_true - y_pred


class DecisionTree:
    def __init__(self, max_depth=3, a_lambda=0.5, gamma=0.1, lead_nodes=0, lead_scores=0, buckets=10):
        """lambda 和 gamma 推荐的初始取值范围为0.1到10"""
        self.max_depth = max_depth
        self.lead_nodes = lead_nodes
        self.lead_scores = lead_scores
        self.a_lambda = a_lambda
        self.gamma = gamma

        self.buckets = buckets
        self.tree = None

    @staticmethod
    def split_data(X, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices

    @staticmethod
    def calculate_leaf_value(gradient):
        return np.mean(gradient)

    def calculate_loss(self, left_gradient, right_gradient):
        return np.sum(left_gradient ** 2) + np.sum(right_gradient ** 2) + self.lead_nodes + self.lead_scores

    @staticmethod
    def is_pure(gradient):
        return np.all([gradient == gradient[0]])

    def predict(self, X):
        return np.array([self.traverse_tree(x) for x in X])

    def traverse_tree(self, x, node=None):
        if node is None:
            node = self.tree

        if "leaf_value" in node:
            return node["leaf_value"]

        feature_index = node["feature_index"]
        threshold = node["threshold"]

        if x[feature_index] <= threshold:
            return self.traverse_tree(x, node["left_child"])
        else:
            return self.traverse_tree(x, node["right_child"])

    def fit(self, X, gradient):
        self.tree = self.build_tree(X, gradient, current_depth=0)

    def build_tree(self, X, gradient, current_depth):
        if current_depth >= self.max_depth or self.is_pure(gradient):
            leaf_value = self.calculate_leaf_value(gradient)
            # 统计叶子节点数目
            self.lead_nodes += self.gamma * 1
            # 计算树的叶子结点分数
            self.lead_scores += self.a_lambda * 0.5 * (leaf_value ** 2)
            return {"leaf_value": leaf_value}

        best_feature_index, best_threshold = self.find_best_split(X, gradient)
        left_indices, right_indices = self.split_data(X, best_feature_index, best_threshold)
        left_gradient = gradient[left_indices]
        right_gradient = gradient[right_indices]

        left_child = self.build_tree(X[left_indices], left_gradient, current_depth + 1)
        right_child = self.build_tree(X[right_indices], right_gradient, current_depth + 1)

        return {
            "feature_index": best_feature_index,
            "threshold": best_threshold,
            "left_child": left_child,
            "right_child": right_child
        }

    def split_buckets(self, X, thresholds):
        """如果特征值是连续的，需要分桶"""
        if len(thresholds) != len(X):
            return thresholds

        max_threshold = int(np.float64(max(thresholds)))
        min_threshold = int(np.float64(min(thresholds)))
        inter_value = math.ceil((max_threshold - min_threshold) / self.buckets)

        return np.array(range(min_threshold, max_threshold, inter_value))

    def find_best_split(self, X, gradient):
        best_feature_index = None
        best_threshold = None
        best_loss = float("inf")

        for feature_index in range(X.shape[1]):

            # 分桶
            thresholds = self.split_buckets(X, np.unique(X[:, feature_index]))

            for threshold in thresholds:
                left_indices, right_indices = self.split_data(X, feature_index, threshold)

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_gradient = gradient[left_indices]
                right_gradient = gradient[right_indices]

                loss = self.calculate_loss(left_gradient, right_gradient)
                if loss < best_loss:
                    best_loss = loss
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold


def create_dataset():
    """
    测试数据
    """
    x_train = np.array([[1, 5.56],
                        [2, 5.70],
                        [3, 5.91],
                        [4, 6.40],
                        [5, 6.80],
                        [6, 7.05],
                        [7, 8.90],
                        [8, 8.70],
                        [9, 9.00],
                        [10, 9.05]])
    x_test = np.array([[2], [5]])
    return x_train, x_test


if __name__ == '__main__':
    train_x, test_x = create_dataset()
    X = train_x[:, :-1]
    y = train_x[:, -1]

    model = XGBoost(n_estimators=100, learning_rate=0.2, max_depth=4)
    model.fit(X, y)
    y_pred = model.predict(test_x)
    print(y_pred)
