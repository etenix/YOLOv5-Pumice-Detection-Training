import torch
import numpy as np
from pathlib import Path

def evaluate_model_performance(model_path, data_config):
    """
    訓練済みモデルの評価を行い、mAP（平均適合率）等の指標を算出する
    """
    # 訓練済みYOLOv5モデルのロード
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    print(f"--- モデル評価開始: {model_path} ---")
    
    # 評価実行 (val.py をラッパーとして使用)
    # 衛星画像特有の微小な軽石を評価するため、conf_thres（信頼度閾値）を調整
    results = model.val(
        data=data_config,
        imgsz=1280,      # 訓練時と同じ高解像度で評価
        conf_thres=0.25, # 検出の閾値
        iou_thres=0.45,  # 重複の閾値
        device='0'
    )
    
    # 指標の抽出
    mp, mr, map50, map95 = results.results_dict['metrics/precision'], \
                           results.results_dict['metrics/recall'], \
                           results.results_dict['metrics/mAP_0.5'], \
                           results.results_dict['metrics/mAP_0.5:0.95']

    print(f"\n評価結果概要:")
    print(f"  Precision (適合率): {mp:.4f}")
    print(f"  Recall (再現率): {mr:.4f}")
    print(f"  mAP@0.5: {map50:.4f} (主要評価指標)")
    print(f"  mAP@0.5:0.95: {map95:.4f}")
    
    return results

if __name__ == "__main__":
    MODEL_PATH = '../weights/best.pt'
    DATA_YAML = '../data/pumice_dataset.yaml'
    
    if Path(MODEL_PATH).exists():
        evaluate_model_performance(MODEL_PATH, DATA_YAML)
    else:
        print("エラー: 訓練済みモデル(best.pt)が見つかりません。")