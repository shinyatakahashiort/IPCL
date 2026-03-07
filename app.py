"""
ICL Size & Vault Prediction App v3
Streamlit application for ICL size and vault prediction using machine learning
"""

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re
import os

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="ICL Size & Vault Predictor",
    page_icon="👁️",
    layout="centered"
)

# ============================================================
# モデル読み込み
# ============================================================
@st.cache_resource
def load_models():
    size_data   = None
    vault_data  = None
    binary_data = None

    if os.path.exists("vault_size_prediction_v3.pkl"):
        with open("vault_size_prediction_v3.pkl", "rb") as f:
            size_data = pickle.load(f)

    if os.path.exists("vault_ml_final.pkl"):
        with open("vault_ml_final.pkl", "rb") as f:
            vault_data = pickle.load(f)

    if os.path.exists("vault_binary_prediction.pkl"):
        with open("vault_binary_prediction.pkl", "rb") as f:
            binary_data = pickle.load(f)

    return size_data, vault_data, binary_data

size_data, vault_data, binary_data = load_models()

# ============================================================
# ユーティリティ関数
# ============================================================
def size_to_nearest(s, candidates):
    return min(candidates, key=lambda x: abs(x - s))

# ============================================================
# サイズ予測（argmaxのみ）
# ============================================================
def predict_size(lv, acd_ratio, cct_cct_acd, acw, sex, data):
    SIZE_CANDIDATES    = data['size_candidates']
    IDX_TO_SIZE        = data['idx_to_size']
    COMPACT_TO_ORIG    = data['compact_to_orig']
    SIZE_ARRAY_PRESENT = data['size_array_present']
    present_classes    = data['present_classes']
    N_CLASSES          = len(present_classes)
    scaler             = data['scaler']
    final_models       = data['final_models']

    X   = np.array([[lv, acd_ratio, cct_cct_acd, acw, sex]])
    X_s = scaler.transform(X)

    all_probs = {}
    for mname, mdl in final_models.items():
        probs = np.zeros(N_CLASSES)
        raw   = mdl.predict_proba(X_s)[0]
        for ci, cls in enumerate(mdl.classes_):
            probs[cls] = raw[ci]
        all_probs[mname] = probs

    ens_prob     = np.mean([all_probs[m] for m in final_models], axis=0)
    pred_compact = int(np.argmax(ens_prob))
    pred_size    = IDX_TO_SIZE[COMPACT_TO_ORIG[pred_compact]]

    prob_display = {}
    for cls in present_classes:
        orig = COMPACT_TO_ORIG[cls]
        size = IDX_TO_SIZE[orig]
        prob_display[size] = float(ens_prob[cls])

    return pred_size, prob_display, SIZE_CANDIDATES

# ============================================================
# Vault予測
# ============================================================
def predict_vault(lv, acd_ratio, cct, acw, pred_size, data):
    feat_orig  = data['feature_names_original']
    scaler     = data['scaler']
    needs_sc   = data['needs_scaling']
    reg_models = data['reg_models']
    thresholds = data['vault_thresholds']
    cat_names  = data['category_names']

    input_map = {
        'LV_SliceNo: 0 (Angle: 180-0)':  lv,
        'ACD[Endo.]_CCT/ACD':             acd_ratio,
        'CCT_SliceNo: 0 (Angle: 180-0)':  cct,
        'ACW_SliceNo: 0 (Angle: 180-0)':  acw,
        'size':                            pred_size,
    }
    X = np.array([[input_map[f] for f in feat_orig]])

    if needs_sc:
        X = scaler.transform(X)

    preds = [float(mdl.predict(X)[0]) for mdl in reg_models.values()]
    vault_pred = float(np.mean(preds))

    vmin = thresholds['min']
    vmax = thresholds['max']
    if vault_pred < vmin:
        cat_idx = 0
    elif vault_pred <= vmax:
        cat_idx = 1
    else:
        cat_idx = 2

    return vault_pred, cat_idx, cat_names[cat_idx], vmin, vmax

# ============================================================
# Vault適正確率予測（二値分類）
# ============================================================
def predict_vault_binary(lv, acd_ratio, cct_cct_acd, acw, pred_size, age, sex, data):
    feat_orig   = data['feature_names_original']
    scaler      = data['scaler']
    top2_names  = data['top2_names']
    # narrow（狭義 400-600µm）モデルを使用
    final_models = data['narrow']['final_models']

    input_map = {
        'LV_SliceNo: 0 (Angle: 180-0)':  lv,
        'ACD[Endo.]_CCT/ACD':             acd_ratio,
        'CCT_CCT/ACD':                    cct_cct_acd,
        'ACW_SliceNo: 0 (Angle: 180-0)':  acw,
        'size':                            pred_size,
        'age':                             age,
        'sex':                             sex,
    }
    X   = np.array([[input_map[f] for f in feat_orig]])
    X_s = scaler.transform(X)

    probs = []
    for mdl in final_models.values():
        p = mdl.predict_proba(X_s)[0]
        pos_idx = list(mdl.classes_).index(1) if 1 in mdl.classes_ else -1
        probs.append(p[pos_idx] if pos_idx >= 0 else 0.5)

    prob_optimal = float(np.mean(probs))
    return prob_optimal

# ============================================================
# UI
# ============================================================
st.title("👁️ ICL Size & Vault Predictor")
st.markdown("**ICLサイズ・Vault予測アプリ** — 眼科測定値からICLサイズとVaultを予測します")
st.markdown("---")

if size_data is None:
    st.error("⚠️ サイズ予測モデル（vault_size_prediction_v3.pkl）が見つかりません。")
    st.stop()

if vault_data is None:
    st.warning("⚠️ Vault予測モデル（vault_ml_final.pkl）が見つかりません。サイズ予測のみ実行します。")

if binary_data is None:
    st.warning("⚠️ Vault適正確率モデル（vault_binary_prediction.pkl）が見つかりません。")

# ============================================================
# 入力フォーム
# ============================================================
st.subheader("📋 測定値入力")

col1, col2 = st.columns(2)

with col1:
    lv = st.number_input(
        "LV (Lens Vault) [mm]",
        min_value=0.0, max_value=2.0, value=0.30, step=0.01, format="%.3f",
        help="LV_SliceNo: 0 (Angle: 180-0)"
    )
    cct_cct_acd = st.number_input(
        "CCT / ACD ratio",
        min_value=0.0, max_value=1.0, value=0.15, step=0.001, format="%.4f",
        help="CCT_CCT/ACD"
    )
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "女性 (0)" if x == 0 else "男性 (1)")

with col2:
    acd_ratio = st.number_input(
        "ACD[Endo.] / (CCT/ACD) ratio",
        min_value=0.0, max_value=10.0, value=3.20, step=0.01, format="%.3f",
        help="ACD[Endo.]_CCT/ACD"
    )
    acw = st.number_input(
        "ACW (Anterior Chamber Width) [mm]",
        min_value=10.0, max_value=14.0, value=11.80, step=0.01, format="%.3f",
        help="ACW_SliceNo: 0 (Angle: 180-0)"
    )
    age = st.number_input(
        "Age（年齢）",
        min_value=10, max_value=80, value=30, step=1,
        help="Vault適正確率予測に使用"
    )

# Vault予測用CCT（vault_ml_finalはCCT_SliceNo列を使用）
if vault_data is not None:
    with st.expander("🔬 Vault予測用追加入力（CCT）"):
        cct_raw = st.number_input(
            "CCT (Central Corneal Thickness) [μm]",
            min_value=400.0, max_value=700.0, value=540.0, step=1.0, format="%.1f",
            help="CCT_SliceNo: 0 (Angle: 180-0) — Vault予測に使用"
        )
else:
    cct_raw = 540.0

st.markdown("---")

# ============================================================
# 予測実行
# ============================================================
if st.button("🔮 予測実行", type="primary", use_container_width=True):

    # サイズ予測
    pred_size, prob_display, SIZE_CANDIDATES = predict_size(
        lv, acd_ratio, cct_cct_acd, acw, sex, size_data
    )

    # Vault予測
    vault_pred = cat_idx = cat_name = None
    if vault_data is not None:
        vault_pred, cat_idx, cat_name, vmin, vmax = predict_vault(
            lv, acd_ratio, cct_raw, acw, pred_size, vault_data
        )

    # Vault適正確率（二値分類）
    prob_optimal = None
    if binary_data is not None:
        prob_optimal = predict_vault_binary(
            lv, acd_ratio, cct_cct_acd, acw, pred_size, age, sex, binary_data
        )

    # ============================================================
    # サイズ予測結果
    # ============================================================
    st.subheader("📊 ICLサイズ予測結果")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric(
            label="🎯 推奨サイズ",
            value=f"{pred_size:.2f} mm",
            help="確率が最も高いサイズ（argmax）"
        )
    with col_s2:
        if prob_optimal is not None:
            color_label = "🟢" if prob_optimal >= 0.6 else "🟡" if prob_optimal >= 0.4 else "🔴"
            st.metric(
                label=f"{color_label} Vault適正確率（狭義: 400-600µm）",
                value=f"{prob_optimal*100:.1f}%",
                help="このサイズでVaultが400-600µmになる予測確率（LightGBM+CatBoost Ensemble, AUC=62.2%）"
            )
            if prob_optimal >= 0.6:
                st.success("✅ Vault適正の可能性が高いです。")
            elif prob_optimal >= 0.4:
                st.warning("⚠️ Vault適正の確率は中程度です。慎重に判断してください。")
            else:
                st.error("❌ Vault不適正のリスクがあります。サイズの再検討を推奨します。")

    # ============================================================
    # Vault予測結果
    # ============================================================
    if vault_pred is not None:
        st.markdown("---")
        st.subheader("🔬 Vault予測結果")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric(
                label="📏 予測Vault値",
                value=f"{vault_pred*1000:.0f} µm",
                delta=f"{vault_pred:.3f} mm",
            )
        with col_v2:
            st.metric(label="📋 Vault判定", value=cat_name)

        if cat_idx == 0:
            st.error(f"⚠️ **Vault低すぎ** ({vault_pred*1000:.0f} µm) — より大きいサイズの検討を推奨します。")
        elif cat_idx == 1:
            st.success(f"✅ **Vault適正** ({vault_pred*1000:.0f} µm) — {pred_size:.2f} mmは適切なサイズと予測されます。")
        else:
            st.error(f"⚠️ **Vault高すぎ** ({vault_pred*1000:.0f} µm) — より小さいサイズの検討を推奨します。")

        # Vaultゲージ
        fig_v, ax_v = plt.subplots(figsize=(8, 1.5))
        ax_v.barh(0, vmin,       color='#EF5350', height=0.4, left=0)
        ax_v.barh(0, vmax-vmin,  color='#66BB6A', height=0.4, left=vmin)
        ax_v.barh(0, 1.0-vmax,   color='#EF5350', height=0.4, left=vmax)
        ax_v.axvline(vault_pred, color='#1565C0', linewidth=3,
                     label=f'予測値: {vault_pred*1000:.0f} µm')
        ax_v.set_xlim(0, 1.0)
        ax_v.set_xlabel('Vault (mm)', fontweight='bold')
        ax_v.set_yticks([])
        ax_v.legend(loc='upper right', fontsize=9)
        ax_v.set_title('Vault 予測値ゲージ', fontweight='bold')
        ax_v.text(vmin/2,        -0.35, 'Low',    ha='center', fontsize=8, color='#B71C1C')
        ax_v.text((vmin+vmax)/2, -0.35, 'Normal', ha='center', fontsize=8, color='#1B5E20')
        ax_v.text((vmax+1.0)/2,  -0.35, 'High',   ha='center', fontsize=8, color='#B71C1C')
        fig_v.tight_layout()
        st.pyplot(fig_v)
        plt.close()

    # ============================================================
    # 臨床参考情報
    # ============================================================
    st.markdown("---")
    st.info(
        "💡 **選択原則**: Vaultが高すぎるリスクを避けるため、迷ったら小さめを選ぶことを推奨します。\n\n"
        "このアプリはあくまで参考情報です。最終的なサイズ選択は医師の判断に基づいてください。"
    )

    # ============================================================
    # 確率分布グラフ
    # ============================================================
    st.markdown("---")
    st.subheader("📈 サイズ確率分布")

    sizes_all = SIZE_CANDIDATES
    probs_all = [prob_display.get(s, 0.0) for s in sizes_all]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#1976D2' if s == pred_size else '#B0BEC5' for s in sizes_all]

    bars = ax.bar(
        [f"{s:.2f}" for s in sizes_all],
        probs_all,
        color=colors, edgecolor='white', linewidth=0.5
    )
    for bar, prob in zip(bars, probs_all):
        if prob > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{prob:.1%}",
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    import matplotlib.patches as mpatches
    patch = mpatches.Patch(color='#1976D2', label=f'予測サイズ: {pred_size:.2f} mm')
    ax.legend(handles=[patch], fontsize=9, loc='upper right')
    ax.set_xlabel('ICL Size (mm)', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Predicted Probability Distribution', fontweight='bold')
    ax.set_ylim(0, max(probs_all) * 1.25 + 0.05)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 確率テーブル
    with st.expander("📋 確率テーブル（全サイズ）"):
        import pandas as pd
        prob_df = pd.DataFrame({
            'Size (mm)':        [f"{s:.2f}" for s in sizes_all],
            'Probability (%)':  [f"{p*100:.1f}" for p in probs_all],
        })
        prob_df = prob_df[prob_df['Probability (%)'] != '0.0'].reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ============================================================
# フッター
# ============================================================
st.markdown("---")
st.caption(
    "Size Model: SVM + LightGBM + RF Ensemble (v3) | "
    "Training LOOCV n=60 (c400_600=1) | "
    "Exact: 60.0%, -0.25mm: 78.3%, AUC: 0.600  |  "
    "Vault Regression: XGBoost + LightGBM Ensemble  |  "
    "Vault Binary: LightGBM + CatBoost Ensemble (AUC: 62.2%)"
)
