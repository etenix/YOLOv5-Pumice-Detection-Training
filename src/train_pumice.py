import os

def train_custom_model():
    """
    軽石検知に最適化したYOLOv5訓練設定
    """
    # 衛星画像内の微小な軽石を検知するため、--img 1280 を推奨
    cmd = (
        "python yolov5/train.py "
        "--img 1280 "           # 小さな対象物のために解像度を上げる
        "--batch 8 "            # 解像度アップに伴いバッチサイズを調整
        "--epochs 150 "         # 収束のためにエポック数を多めに設定
        "--data data/pumice_dataset.yaml "
        "--weights yolov5s.pt " 
        "--hyp data/hyps/hyp.scratch-low.yaml " # 衛星画像用のハイパーパラメータ
        "--device 0"
    )
    
    print("軽石検知モデルの訓練を開始します...")
    os.system(cmd)

if __name__ == "__main__":
    train_custom_model()