"""
YOLOv5-Pumice-Detection-Training
衛星画像を用いた軽石（Pumice）検知モデルの訓練・評価パッケージ
"""

from .train_pumice import train_custom_model
from .spectral_filter import apply_pumice_index
from .evaluate_map import evaluate_model_performance

__version__ = "1.0.0"
__author__ = "孫 浩然 (Sun Haoran)"
__description__ = "分光反射特性を活用した軽石特化型物体検知モデルの訓練パイプライン"

# 必要に応じて、訓練開始時のGPUチェックやディレクトリの自動作成ロジックをここに記述します。