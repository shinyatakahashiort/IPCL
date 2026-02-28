"""
ICL Size & Vault Prediction App
Streamlit application for ICL size and vault prediction using machine learning
"""

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    size_data  = None
    vault_data = None

    if os.path.exists("vault_size_prediction.pkl"):
        with open("vault_size_prediction.pkl", "rb") as f:
            size_data = pickle.load(f)

    if os.path.exists("vault_ml_final.pkl"):
        with open("vault_ml_final.pkl", "rb") as f:
            vault_data = pickle.load(f)

    return size_data, vault_data

size_data, vault_data = load_models()

# ============================================================
# ユーティリティ関数
# ============================================================
def clean_feature_name(name):
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

def size_to_nearest(s, candidates):
    return min(candidates, key=lambda x: abs(x - s))

# ============================================================
# サイズ予測
# ============================================================
def predict_size(lv, acd_ratio, cct, acw, data):
    SIZE_CANDIDATES    = data['size_candidates']
    IDX_TO_SIZE        = data['idx_to_size']
    ORIG_TO_COMPACT    = data['orig_to_compact']
    COMPACT_TO_ORIG    = data['compact_to_orig']
    SIZE_ARRAY_PRESENT = data['size_array_present']
    present_classes    = data['present_classes']
    N_CLASSES          = len(present_classes)
    scaler             = data['scaler']
    final_models       = data['final_models']

    X   = np.array([[lv, acd_ratio, cct, acw]])
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

    weighted  = float(ens_prob @ SIZE_ARRAY_PRESENT)
    pred_wavg = size_to_nearest(weighted, SIZE_CANDIDATES)

    prob_display = {}
    for cls in present_classes:
        orig = COMPACT_TO_ORIG[cls]
        size = IDX_TO_SIZE[orig]
        prob_display[size] = float(ens_prob[cls])

    return pred_size, pred_wavg, weighted, prob_display, SIZE_CANDIDATES

# ============================================================
# Vault予測
# ============================================================
def predict_vault(lv, acd_ratio, cct, acw, pred_size, data):
    feat_orig   = data['feature_names_original']
    feat_clean  = data['feature_names_clean']
    scaler      = data['scaler']
    needs_sc    = data['needs_scaling']
    reg_models  = data['reg_models']
    thresholds  = data['vault_thresholds']
    cat_names   = data['category_names']

    # 入力値をfeature_names_originalの順に並べる
    input_map = {
        'LV_SliceNo: 0 (Angle: 180-0)':       lv,
        'ACD[Endo.]_CCT/ACD':                  acd_ratio,
        'CCT_SliceNo: 0 (Angle: 180-0)':       cct,
        'ACW_SliceNo: 0 (Angle: 180-0)':       acw,
        'size':                                 pred_size,
    }
    X = np.array([[input_map[f] for f in feat_orig]])

    if needs_sc:
        X = scaler.transform(X)

    # アンサンブル予測（平均）
    preds = []
    for mname, mdl in reg_models.items():
        preds.append(float(mdl.predict(X)[0]))
    vault_pred = float(np.mean(preds))

    # カテゴリ判定
    vmin = thresholds['min']  # 0.25mm
    vmax = thresholds['max']  # 0.75mm
    if vault_pred < vmin:
        cat_idx = 0
    elif vault_pred <= vmax:
        cat_idx = 1
    else:
        cat_idx = 2
    cat_name = cat_names[cat_idx]

    return vault_pred, cat_idx, cat_name, vmin, vmax

# ============================================================
# UI
# ============================================================
st.title("👁️ ICL Size & Vault Predictor")
st.markdown("**ICLサイズ・Vault予測アプリ** — 4つの眼科測定値からICLサイズとVaultを予測します")
st.markdown("---")

if size_data is None:
    st.error("⚠️ サイズ予測モデル（vault_size_prediction.pkl）が見つかりません。")
    st.stop()

if vault_data is None:
    st.warning("⚠️ Vault予測モデル（vault_ml_final.pkl）が見つかりません。サイズ予測のみ実行します。")

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
    cct = st.number_input(
        "CCT (Central Corneal Thickness) [μm]",
        min_value=400.0, max_value=700.0, value=540.0, step=1.0, format="%.1f",
        help="CCT_SliceNo: 0 (Angle: 180-0)"
    )

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

st.markdown("---")

# ============================================================
# 予測実行
# ============================================================
if st.button("🔮 予測実行", type="primary", use_container_width=True):

    # サイズ予測
    pred_size, pred_wavg, weighted_raw, prob_display, SIZE_CANDIDATES = predict_size(
        lv, acd_ratio, cct, acw, size_data
    )

    # Vault予測
    vault_pred = None
    cat_idx    = None
    cat_name   = None
    if vault_data is not None:
        vault_pred, cat_idx, cat_name, vmin, vmax = predict_vault(
            lv, acd_ratio, cct, acw, pred_size, vault_data
        )

    # ============================================================
    # サイズ予測結果
    # ============================================================
    st.subheader("📊 ICLサイズ予測結果")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="🎯 推奨サイズ（argmax）",
            value=f"{pred_size:.2f} mm",
            help="最も確率が高いサイズ"
        )
    with col_b:
        st.metric(
            label="⚖️ 加重平均サイズ（wavg）",
            value=f"{pred_wavg:.2f} mm",
            delta=f"連続値: {weighted_raw:.3f} mm",
            help="確率分布の加重平均から算出"
        )

    if pred_size == pred_wavg:
        st.success(f"✅ argmax・加重平均ともに **{pred_size:.2f} mm** で一致。信頼性が高い予測です。")
    else:
        st.warning(
            f"⚠️ argmax（{pred_size:.2f} mm）と加重平均（{pred_wavg:.2f} mm）が異なります。"
            f"確率が拮抗しています。臨床判断を優先してください。"
        )

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
                help="予測されるVault値"
            )
        with col_v2:
            st.metric(
                label="📋 Vault判定",
                value=cat_name,
                help=f"Low: <{vmin*1000:.0f}µm / Normal: {vmin*1000:.0f}-{vmax*1000:.0f}µm / High: >{vmax*1000:.0f}µm"
            )

        # カテゴリ別カラー表示
        if cat_idx == 0:
            st.error(f"⚠️ **Vault低すぎ** ({vault_pred*1000:.0f} µm) — より大きいサイズの検討を推奨します。")
        elif cat_idx == 1:
            st.success(f"✅ **Vault適正** ({vault_pred*1000:.0f} µm) — {pred_size:.2f} mmは適切なサイズと予測されます。")
        else:
            st.error(f"⚠️ **Vault高すぎ** ({vault_pred*1000:.0f} µm) — より小さいサイズの検討を推奨します。")

        # Vaultゲージ
        fig_v, ax_v = plt.subplots(figsize=(8, 1.5))
        ax_v.barh(0, vmin, color='#EF5350', height=0.4, left=0)
        ax_v.barh(0, vmax - vmin, color='#66BB6A', height=0.4, left=vmin)
        ax_v.barh(0, 1.0 - vmax, color='#EF5350', height=0.4, left=vmax)
        ax_v.axvline(vault_pred, color='#1565C0', linewidth=3, label=f'予測値: {vault_pred*1000:.0f} µm')
        ax_v.set_xlim(0, 1.0)
        ax_v.set_xlabel('Vault (mm)', fontweight='bold')
        ax_v.set_yticks([])
        ax_v.legend(loc='upper right', fontsize=9)
        ax_v.set_title('Vault 予測値ゲージ', fontweight='bold')
        ax_v.text(vmin/2, -0.35, 'Low', ha='center', fontsize=8, color='#B71C1C')
        ax_v.text((vmin+vmax)/2, -0.35, 'Normal', ha='center', fontsize=8, color='#1B5E20')
        ax_v.text((vmax+1.0)/2, -0.35, 'High', ha='center', fontsize=8, color='#B71C1C')
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
    colors = []
    for s in sizes_all:
        if s == pred_size:
            colors.append('#1976D2')
        elif s == pred_wavg:
            colors.append('#F57C00')
        else:
            colors.append('#B0BEC5')

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

    patch_am = mpatches.Patch(color='#1976D2', label=f'argmax: {pred_size:.2f} mm')
    patch_wv = mpatches.Patch(color='#F57C00', label=f'wavg: {pred_wavg:.2f} mm')
    ax.legend(handles=[patch_am, patch_wv], fontsize=9, loc='upper right')
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
            'Size (mm)': [f"{s:.2f}" for s in sizes_all],
            'Probability (%)': [f"{p*100:.1f}" for p in probs_all],
        })
        prob_df = prob_df[prob_df['Probability (%)'] != '0.0'].reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ============================================================
# フッター
# ============================================================
st.markdown("---")
st.caption(
    "Size Model: RF + CatBoost Ensemble | Training LOOCV n=44 (c400_600=1) | "
    "Exact: 54.5%, -0.25mm: 75.0%, AUC: 0.723  |  "
    "Vault Model: XGBoost + LightGBM Ensemble"
)
