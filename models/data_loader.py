class ModelConfig:
    """
    模型超参数配置类
    对应中期报告中的参数设置 [cite: 244, 267]
    """
    def __init__(self):
        # 数据路径
        self.data_path = './data/corpus_sample.csv'
        self.vocab_path = './models/vocab.txt'
        
        # 主题模型参数
        self.K = 20                # 隐主题数 (Topic Number) [cite: 267]
        self.alpha = 0.1           # Dirichlet先验参数 alpha [cite: 267]
        self.beta = 0.01           # Dirichlet先验参数 beta
        
        # 训练参数
        self.n_iter = 5000         # 总迭代次数 [cite: 247]
        self.burn_in = 1000        # 燃烧期 (Burn-in) [cite: 244]
        self.sample_interval = 50  # 采样间隔 [cite: 244]
        self.batch_size = 64
        
        # 深度学习参数
        self.bert_model = 'bert-base-chinese' # [cite: 242]
        self.embed_dim = 768       # BERT向量维度