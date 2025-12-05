import argparse
from config import ModelConfig
from models.data_loader import AutomotiveDataLoader
from models.em_trainer import MsSTMTrainer

def main():
    parser = argparse.ArgumentParser(description='MsSTM for Vehicle Defect Detection')
    parser.add_argument('--mode', type=str, default='train', help='train or predict')
    args = parser.parse_args()

    # 加载配置 [cite: 244]
    config = ModelConfig()
    
    # 数据加载 [cite: 245]
    loader = AutomotiveDataLoader(config)
    
    # 初始化训练器
    trainer = MsSTMTrainer(config)
    
    if args.mode == 'train':
        trainer.train(loader)
    else:
        print("Prediction mode...")

if __name__ == '__main__':
    main()