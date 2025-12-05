import tensorflow as tf
from sklearn.linear_model import LogisticRegression

class RecallPredictor:
    """
    下游监督子模型 (Downstream Supervision Sub-model)。
    利用逻辑回归生成响应变量（是否召回）[cite: 237]。
    """
    def __init__(self, config):
        self.config = config
        # 使用L-BFGS优化器 [cite: 247]
        self.model = LogisticRegression(solver='lbfgs', max_iter=200)

    def optimize(self, topic_distribution, labels):
        """
        M-Step: 优化回归系数
        """
        print("M-Step: Optimizing parameters using L-BFGS...")
        self.model.fit(topic_distribution, labels)

    def predict_risk(self, topic_distribution, metadata_vector):
        """
        输入：主题分布 + 元数据向量 (32维)
        输出：召回风险概率 (ROC-AUC) [cite: 247]
        """
        # 特征融合
        features = np.concatenate([topic_distribution, metadata_vector], axis=1)
        risk_score = self.model.predict_proba(features)[:, 1]
        return risk_score