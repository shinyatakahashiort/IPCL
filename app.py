"""
ICL Size & Vault Prediction App v3
Streamlit application for ICL size and vault prediction using machine learning
"""

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
# サイズ予測
# ============================================================
def predict_size(lv, acd, cct_um, acw, sex, data):
    SIZE_CANDIDATES = data['size_candidates']
    IDX_TO_SIZE     = data['idx_to_size']
    COMPACT_TO_ORIG = data['compact_to_orig']
    present_classes = data['present_classes']
    N_CLASSES       = len(present_classes)
    scaler          = data['scaler']
    final_models    = data['final_models']

    # モデル特徴量: LV, ACD(mm), CCT(μm), ACW, sex
    X   = np.array([[lv, acd, cct_um, acw, sex]])
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
def predict_vault(lv, acd, cct_um, acw, pred_size, age, sex, data):
    feat_orig  = data['feature_names_original']
    scaler     = data['scaler']
    needs_sc   = data['needs_scaling']
    model      = data['regression_model']
    thresholds = data['vault_thresholds']
    cat_names  = data['category_names']

    # モデル特徴量: LV, ACD(mm), CCT(μm), size, ACW, age, sex
    input_map = {
        'LV_SliceNo: 0 (Angle: 180-0)':  lv,
        'ACD[Endo.]_CCT/ACD':             acd,
        'CCT_CCT/ACD':                    cct_um,
        'ACW_SliceNo: 0 (Angle: 180-0)':  acw,
        'size':                            pred_size,
        'age':                             age,
        'sex':                             sex,
    }
    X = np.array([[input_map[f] for f in feat_orig]])

    if needs_sc:
        X = scaler.transform(X)

    vault_pred = float(model.predict(X)[0])

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
def predict_vault_binary(lv, acd, cct_um, acw, pred_size, age, sex, data):
    feat_orig    = data['feature_names_original']
    scaler       = data['scaler']
    final_models = data['final_models']

    input_map = {
        'LV_SliceNo: 0 (Angle: 180-0)':  lv,
        'ACD[Endo.]_CCT/ACD':             acd,
        'CCT_CCT/ACD':                    cct_um,
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

    return float(np.mean(probs))

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
        min_value=-1.0, max_value=1.0, value=0.03, step=0.01, format="%.2f"
    )
    cct_um = st.number_input(
        "CCT (Central Corneal Thickness) [µm]",
        min_value=400, max_value=700, value=530, step=1
    )
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "女性 (0)" if x == 0 else "男性 (1)")

with col2:
    acd = st.number_input(
        "ACD [Endo.] (Anterior Chamber Depth) [mm]",
        min_value=2.0, max_value=5.0, value=3.08, step=0.01, format="%.2f"
    )
    acw = st.number_input(
        "ACW (Anterior Chamber Width) [mm]",
        min_value=10.0, max_value=14.0, value=11.70, step=0.01, format="%.2f"
    )
    age = st.number_input(
        "Age（年齢）",
        min_value=10, max_value=80, value=44, step=1
    )

st.markdown("---")

# ============================================================
# 予測実行
# ============================================================
if st.button("🔮 予測実行", type="primary", use_container_width=True):

    # サイズ予測
    pred_size, prob_display, SIZE_CANDIDATES = predict_size(
        lv, acd, cct_um, acw, sex, size_data
    )

    # 推奨・-0.25mm・-0.50mmの3サイズを計算
    candidates = [pred_size, pred_size - 0.25, pred_size - 0.50]
    size_results = []

    for s in candidates:
        vault_val = None
        vault_cat = None
        vault_cat_name = None
        vmin = vmax = None
        prob_opt = None

        if vault_data is not None:
            vault_val, vault_cat, vault_cat_name, vmin, vmax = predict_vault(
                lv, acd, cct_um, acw, s, age, sex, vault_data
            )
        if binary_data is not None:
            prob_opt = predict_vault_binary(
                lv, acd, cct_um, acw, s, age, sex, binary_data
            )

        size_results.append({
            'size': s,
            'vault': vault_val,
            'vault_cat': vault_cat,
            'vault_cat_name': vault_cat_name,
            'prob_opt': prob_opt,
            'vmin': vmin,
            'vmax': vmax,
        })

    # ============================================================
    # 推奨サイズ表示
    # ============================================================
    st.subheader("📊 ICLサイズ予測結果")
    st.metric(label="🎯 推奨サイズ（argmax）", value=f"{pred_size:.2f} mm")

    # ============================================================
    # サイズ比較テーブル
    # ============================================================
    st.markdown("---")
    st.subheader("🔬 サイズ別 Vault・適正確率")

    labels = ["推奨サイズ", "推奨 −0.25mm", "推奨 −0.50mm"]
    cols = st.columns(3)

    for col, label, r in zip(cols, labels, size_results):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(f"### {r['size']:.2f} mm")

            if r['vault'] is not None:
                vault_um = r['vault'] * 1000
                cat = r['vault_cat']
                if cat == 0:
                    color = "🔵"
                elif cat == 1:
                    color = "🟢"
                else:
                    color = "🔴"
                st.metric("予測Vault", f"{vault_um:.0f} µm")
                st.markdown(f"{color} **{r['vault_cat_name']}**")
            else:
                st.markdown("Vault: —")

            if r['prob_opt'] is not None:
                p = r['prob_opt']
                emoji = "🟢" if p >= 0.6 else "🟡" if p >= 0.4 else "🔴"
                st.metric("Vault適正確率", f"{p*100:.1f}%")
                st.markdown(f"{emoji} {'高' if p >= 0.6 else '中' if p >= 0.4 else '低'}")
            else:
                st.markdown("Vault適正確率: —")

    # ============================================================
    # Vaultゲージ（推奨サイズのみ）
    # ============================================================
    r0 = size_results[0]
    if r0['vault'] is not None:
        vmin, vmax = r0['vmin'], r0['vmax']
        fig_v, ax_v = plt.subplots(figsize=(8, 1.5))
        ax_v.barh(0, vmin,       color='#EF5350', height=0.4, left=0)
        ax_v.barh(0, vmax-vmin,  color='#66BB6A', height=0.4, left=vmin)
        ax_v.barh(0, 1.0-vmax,   color='#EF5350', height=0.4, left=vmax)
        for r, ls, lw in zip(size_results, ['-', '--', ':'], [3, 2, 2]):
            if r['vault'] is not None:
                ax_v.axvline(r['vault'], color='#1565C0', linewidth=lw, linestyle=ls,
                             label=f"{r['size']:.2f}mm: {r['vault']*1000:.0f}µm")
        ax_v.set_xlim(0, 1.0)
        ax_v.set_xlabel('Vault (mm)', fontweight='bold')
        ax_v.set_yticks([])
        ax_v.legend(loc='upper right', fontsize=8)
        ax_v.set_title('Vault 予測値ゲージ（実線=推奨, 破線=-0.25, 点線=-0.50）', fontweight='bold')
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
