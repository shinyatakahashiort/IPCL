# ICL Size Predictor

機械学習を用いたICL（Intraocular Collamer Lens）サイズ予測アプリです。

## 概要

4つの眼科測定値を入力することで、最適なICLサイズを予測します。

**予測モデル**: Random Forest + CatBoost アンサンブル  
**学習データ**: c400_600=1（Vault適正）44例  
**評価**: LOOCV Exact=54.5%, -0.25mm=75.0%, AUC=0.723

## 入力特徴量

| 特徴量 | 説明 |
|--------|------|
| LV (Lens Vault) | LV_SliceNo: 0 (Angle: 180-0) |
| ACD[Endo.] / (CCT/ACD) ratio | ACD[Endo.]_CCT/ACD |
| CCT (Central Corneal Thickness) | CCT_SliceNo: 0 (Angle: 180-0) |
| ACW (Anterior Chamber Width) | ACW_SliceNo: 0 (Angle: 180-0) |

## 使い方

### ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud デプロイ手順

1. このリポジトリをGitHubにpush
2. `vault_size_prediction.pkl` をリポジトリのルートに配置
3. [Streamlit Cloud](https://streamlit.io/cloud) でリポジトリを連携
4. `app.py` を指定してデプロイ

## ファイル構成

```
├── app.py                      # Streamlitアプリ本体
├── requirements.txt            # 依存パッケージ
├── vault_size_prediction.pkl   # 訓練済みモデル（要配置）
└── README.md
```

## 注意事項

- 本アプリはあくまで参考情報です。最終的なサイズ選択は医師の判断に基づいてください。
- 訓練データに存在しないサイズ（11.00mm、13.25mm以上）は予測対象外です。
- Vaultが高すぎるリスクを避けるため、迷ったら小さめを選ぶことを推奨します。
