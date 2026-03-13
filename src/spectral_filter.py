import numpy as np
import cv2

def apply_pumice_index(band_data):
    """
    軽石の分光反射特性に基づき、特定バンド間で正規化を行い、
    軽石の存在可能性が高い領域を強調する前処理。
    """
    # 例: 近赤外(NIR)と短波赤外(SWIR)を用いた正規化指標の算出
    # (NIR - SWIR) / (NIR + SWIR) のような演算
    nir = band_data['NIR']
    swir = band_data['SWIR']
    pumice_index = (nir - swir) / (nir + swir + 1e-6)
    
    # 0-255のグレースケール画像に変換してAIモデルへ入力
    pumice_index_normalized = cv2.normalize(pumice_index, None, 0, 255, cv2.NORM_MINMAX)
    return pumice_index_normalized.astype(np.uint8)