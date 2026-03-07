"""
ICL Size & Vault Prediction App v3
Streamlit application for ICL size and vault prediction using machine learning
"""

import streamlit as st
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import datetime
from io import BytesIO

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="IPCLsize",
    page_icon="👁️",
    layout="wide"
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
# 予測関数
# ============================================================
def predict_size(lv, acd, cct_um, acw, sex, data):
    SIZE_CANDIDATES = data['size_candidates']
    IDX_TO_SIZE     = data['idx_to_size']
    COMPACT_TO_ORIG = data['compact_to_orig']
    present_classes = data['present_classes']
    N_CLASSES       = len(present_classes)
    scaler          = data['scaler']
    final_models    = data['final_models']

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


def predict_vault(lv, acd, cct_um, acw, pred_size, age, sex, data):
    feat_orig  = data['feature_names_original']
    scaler     = data['scaler']
    needs_sc   = data['needs_scaling']
    model      = data['regression_model']
    thresholds = data['vault_thresholds']
    cat_names  = data['category_names']

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
    cat_idx = 0 if vault_pred < vmin else (1 if vault_pred <= vmax else 2)

    return vault_pred, cat_idx, cat_names[cat_idx], vmin, vmax


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


def run_all_predictions(lv, acd, cct_um, acw, sex, age):
    pred_size, prob_display, SIZE_CANDIDATES = predict_size(
        lv, acd, cct_um, acw, sex, size_data
    )
    candidates = [pred_size, pred_size - 0.25, pred_size - 0.50]
    size_results = []
    for s in candidates:
        vault_val = vault_cat = vault_cat_name = vmin = vmax = prob_opt = None
        if vault_data is not None:
            vault_val, vault_cat, vault_cat_name, vmin, vmax = predict_vault(
                lv, acd, cct_um, acw, s, age, sex, vault_data
            )
        if binary_data is not None:
            prob_opt = predict_vault_binary(
                lv, acd, cct_um, acw, s, age, sex, binary_data
            )
        size_results.append({
            'size': s, 'vault': vault_val, 'vault_cat': vault_cat,
            'vault_cat_name': vault_cat_name, 'prob_opt': prob_opt,
            'vmin': vmin, 'vmax': vmax,
        })
    return pred_size, prob_display, SIZE_CANDIDATES, size_results

# ============================================================
# チャート生成（UI & PDF 共用）
# ============================================================
def make_gauge_fig(size_results, figsize=(6, 1.4)):
    r0 = size_results[0]
    if r0['vault'] is None:
        return None
    vmin, vmax = r0['vmin'], r0['vmax']
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(0, vmin,      color='#EF5350', height=0.4, left=0)
    ax.barh(0, vmax-vmin, color='#66BB6A', height=0.4, left=vmin)
    ax.barh(0, 1.0-vmax,  color='#EF5350', height=0.4, left=vmax)
    for r, ls, lw in zip(size_results, ['-', '--', ':'], [3, 2, 2]):
        if r['vault'] is not None:
            ax.axvline(r['vault'], color='#1565C0', linewidth=lw, linestyle=ls,
                       label=f"{r['size']:.2f} mm: {r['vault']*1000:.0f} um")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Vault (mm)', fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Vault Gauge  (solid=rec, dashed=-0.25, dotted=-0.50)',
                 fontweight='bold', fontsize=9)
    ax.text(vmin/2,        -0.35, 'Low',    ha='center', fontsize=7, color='#B71C1C')
    ax.text((vmin+vmax)/2, -0.35, 'Normal', ha='center', fontsize=7, color='#1B5E20')
    ax.text((vmax+1.0)/2,  -0.35, 'High',   ha='center', fontsize=7, color='#B71C1C')
    fig.tight_layout()
    return fig


def make_dist_fig(pred_size, prob_display, SIZE_CANDIDATES, figsize=(7, 3)):
    sizes_all = SIZE_CANDIDATES
    probs_all = [prob_display.get(s, 0.0) for s in sizes_all]
    fig, ax = plt.subplots(figsize=figsize)
    colors_bar = ['#1976D2' if s == pred_size else '#B0BEC5' for s in sizes_all]
    bars = ax.bar([f"{s:.2f}" for s in sizes_all], probs_all,
                  color=colors_bar, edgecolor='white', linewidth=0.5)
    for bar, prob in zip(bars, probs_all):
        if prob > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{prob:.1%}", ha='center', va='bottom', fontsize=7, fontweight='bold')
    patch = mpatches.Patch(color='#1976D2', label=f'Predicted: {pred_size:.2f} mm')
    ax.legend(handles=[patch], fontsize=8, loc='upper right')
    ax.set_xlabel('ICL Size (mm)', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Predicted Probability Distribution', fontweight='bold')
    ax.set_ylim(0, max(probs_all) * 1.25 + 0.05)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

# ============================================================
# PDF生成
# ============================================================
def generate_pdf(sex, age,
                 inputs_od, pred_od, size_results_od, prob_display_od, size_cands_od,
                 inputs_os, pred_os, size_results_os, prob_display_os, size_cands_os):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer, Image as RLImage,
                                     HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=16,
                         spaceAfter=4, alignment=TA_CENTER)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=12,
                         spaceAfter=3, spaceBefore=6)
    normal = styles['Normal']
    elements = []

    # ---- タイトル ----
    elements.append(Paragraph("IPCLsize Report", h1))
    elements.append(Paragraph(
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}",
        ParagraphStyle('center', parent=normal, alignment=TA_CENTER, fontSize=9)
    ))
    elements.append(Spacer(1, 4*mm))

    # ---- 患者情報 ----
    sex_str = "Male" if sex == 0 else "Female"
    pt_data = [["Sex", sex_str, "Age", f"{age} yrs"]]
    pt_tbl = Table(pt_data, colWidths=[25*mm, 55*mm, 25*mm, 55*mm])
    pt_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), rl_colors.lightgrey),
        ('BACKGROUND', (2, 0), (2, 0), rl_colors.lightgrey),
        ('FONTNAME',   (0, 0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1,-1), 10),
        ('GRID',       (0, 0), (-1,-1), 0.5, rl_colors.grey),
        ('ALIGN',      (0, 0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1,-1), 4),
        ('BOTTOMPADDING', (0, 0), (-1,-1), 4),
    ]))
    elements.append(pt_tbl)
    elements.append(Spacer(1, 5*mm))

    header_colors = {'OD': rl_colors.HexColor('#1565C0'),
                     'OS': rl_colors.HexColor('#1B5E20')}

    for eye_label, key, inputs, pred_size, size_results, prob_display, SIZE_CANDIDATES in [
        ("OD  (Right Eye)", "OD", inputs_od, pred_od, size_results_od, prob_display_od, size_cands_od),
        ("OS  (Left Eye)",  "OS", inputs_os, pred_os, size_results_os, prob_display_os, size_cands_os),
    ]:
        elements.append(HRFlowable(width="100%", thickness=1.5,
                                    color=header_colors[key], spaceAfter=3))
        elements.append(Paragraph(eye_label, h2))

        # -- 測定値 --
        meas_data = [
            ["LV",  f"{inputs['lv']:.2f} mm",
             "CCT", f"{inputs['cct_um']:.0f} um"],
            ["ACD", f"{inputs['acd']:.2f} mm",
             "ACW", f"{inputs['acw']:.2f} mm"],
        ]
        meas_tbl = Table(meas_data, colWidths=[20*mm, 50*mm, 20*mm, 50*mm])
        meas_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0,-1), rl_colors.HexColor('#E3F2FD')),
            ('BACKGROUND', (2, 0), (2,-1), rl_colors.HexColor('#E3F2FD')),
            ('FONTNAME',   (0, 0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0, 0), (-1,-1), 9),
            ('GRID',       (0, 0), (-1,-1), 0.5, rl_colors.grey),
            ('ALIGN',      (0, 0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0, 0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1,-1), 3),
            ('BOTTOMPADDING', (0, 0), (-1,-1), 3),
        ]))
        elements.append(meas_tbl)
        elements.append(Spacer(1, 3*mm))

        # -- 推奨サイズ --
        elements.append(Paragraph(
            f"<b>Recommended Size: {pred_size:.2f} mm</b>",
            ParagraphStyle('rec', parent=normal, fontSize=11)
        ))
        elements.append(Spacer(1, 2*mm))

        # -- サイズ比較テーブル --
        result_header = ["Size", "Predicted Vault", "Vault Status", "Vault Prob."]
        result_rows = [result_header]
        for i, r in enumerate(size_results):
            label = ["Recommended", "-0.25 mm", "-0.50 mm"][i]
            vault_str = f"{r['vault']*1000:.0f} um" if r['vault'] is not None else "-"
            cat_str   = r['vault_cat_name'] if r['vault_cat_name'] else "-"
            prob_str  = f"{r['prob_opt']*100:.1f}%" if r['prob_opt'] is not None else "-"
            result_rows.append([f"{r['size']:.2f} mm  ({label})",
                                 vault_str, cat_str, prob_str])

        res_tbl = Table(result_rows, colWidths=[50*mm, 35*mm, 35*mm, 30*mm])
        res_style = [
            ('BACKGROUND',    (0, 0), (-1, 0), rl_colors.HexColor('#37474F')),
            ('TEXTCOLOR',     (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME',      (0, 1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0, 0), (-1,-1), 9),
            ('GRID',          (0, 0), (-1,-1), 0.5, rl_colors.grey),
            ('ALIGN',         (0, 0), (-1,-1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',    (0, 0), (-1,-1), 4),
            ('BOTTOMPADDING', (0, 0), (-1,-1), 4),
            # 推奨行を青背景
            ('BACKGROUND', (0, 1), (-1, 1), rl_colors.HexColor('#E3F2FD')),
        ]
        # Vault判定で色付け
        for i, r in enumerate(size_results):
            row = i + 1
            if r['vault_cat'] == 1:
                res_style.append(('TEXTCOLOR', (2, row), (2, row),
                                   rl_colors.HexColor('#1B5E20')))
            elif r['vault_cat'] in (0, 2):
                res_style.append(('TEXTCOLOR', (2, row), (2, row),
                                   rl_colors.HexColor('#B71C1C')))
        res_tbl.setStyle(TableStyle(res_style))
        elements.append(res_tbl)
        elements.append(Spacer(1, 3*mm))

        # -- Vaultゲージ --
        fig_g = make_gauge_fig(size_results, figsize=(7, 1.6))
        if fig_g:
            img_buf = fig_to_png_bytes(fig_g)
            plt.close(fig_g)
            elements.append(RLImage(img_buf, width=160*mm, height=36*mm))
        elements.append(Spacer(1, 2*mm))

        # -- 確率分布 --
        fig_d = make_dist_fig(pred_size, prob_display, SIZE_CANDIDATES, figsize=(7, 2.8))
        img_buf2 = fig_to_png_bytes(fig_d)
        plt.close(fig_d)
        elements.append(RLImage(img_buf2, width=160*mm, height=64*mm))
        elements.append(Spacer(1, 4*mm))

    # ---- フッター ----
    elements.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_colors.grey, spaceAfter=3))
    elements.append(Paragraph(
        "This report is for reference only. Final decisions should be based on physician judgment.  |  IPCLsize",
        ParagraphStyle('footer', parent=normal, fontSize=7,
                        alignment=TA_CENTER, textColor=rl_colors.grey)
    ))

    doc.build(elements)
    return buf.getvalue()

# ============================================================
# UI表示ヘルパー
# ============================================================
def show_eye_results(pred_size, prob_display, SIZE_CANDIDATES, size_results):
    st.metric(label="🎯 Recommended Size", value=f"{pred_size:.2f} mm")
    st.markdown("---")

    labels = ["Recommended", "Rec. −0.25mm", "Rec. −0.50mm"]
    cols = st.columns(3)
    for col, label, r in zip(cols, labels, size_results):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(f"### {r['size']:.2f} mm")
            if r['vault'] is not None:
                cat = r['vault_cat']
                color = "🔵" if cat == 0 else "🟢" if cat == 1 else "🔴"
                st.metric("Predicted Vault", f"{r['vault']*1000:.0f} µm")
                st.markdown(f"{color} **{r['vault_cat_name']}**")
            else:
                st.markdown("Vault: —")
            if r['prob_opt'] is not None:
                p = r['prob_opt']
                emoji = "🟢" if p >= 0.6 else "🟡" if p >= 0.4 else "🔴"
                st.metric("Vault Prob.", f"{p*100:.1f}%")
                st.markdown(f"{emoji} {'High' if p >= 0.6 else 'Mid' if p >= 0.4 else 'Low'}")

    fig_g = make_gauge_fig(size_results)
    if fig_g:
        st.pyplot(fig_g)
        plt.close(fig_g)

    fig_d = make_dist_fig(pred_size, prob_display, SIZE_CANDIDATES)
    st.pyplot(fig_d)
    plt.close(fig_d)

    with st.expander("Probability Table"):
        import pandas as pd
        sizes_all = SIZE_CANDIDATES
        probs_all = [prob_display.get(s, 0.0) for s in sizes_all]
        prob_df = pd.DataFrame({
            'Size (mm)':       [f"{s:.2f}" for s in sizes_all],
            'Probability (%)': [f"{p*100:.1f}" for p in probs_all],
        })
        prob_df = prob_df[prob_df['Probability (%)'] != '0.0'].reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


# ============================================================
# UI
# ============================================================
st.title("👁️ IPCLsize")
st.markdown("**IPCLsize** — 眼科測定値からICLサイズとVaultを予測します")
st.markdown("---")

if size_data is None:
    st.error("⚠️ サイズ予測モデル（vault_size_prediction_v3.pkl）が見つかりません。")
    st.stop()
if vault_data is None:
    st.warning("⚠️ Vault予測モデル（vault_ml_final.pkl）が見つかりません。")
if binary_data is None:
    st.warning("⚠️ Vault適正確率モデル（vault_binary_prediction.pkl）が見つかりません。")

# ---- 患者情報 ----
st.subheader("👤 Patient Info")
pc1, pc2 = st.columns(2)
with pc1:
    sex = st.selectbox("Sex", options=[0, 1],
                       format_func=lambda x: "男性 (0)" if x == 0 else "女性 (1)")
with pc2:
    age = st.number_input("Age", min_value=10, max_value=80, value=44, step=1)

st.markdown("---")

# ---- 両眼入力 ----
st.subheader("📋 Measurements")
col_od, col_os = st.columns(2)

with col_od:
    st.markdown("### 👁 OD (右眼)")
    lv_od  = st.number_input("LV [mm]",  min_value=-1.0, max_value=1.0,  value=0.03, step=0.01, format="%.2f", key="lv_od")
    cct_od = st.number_input("CCT [µm]", min_value=400,  max_value=700,  value=530,  step=1,                   key="cct_od")
    acd_od = st.number_input("ACD [mm]", min_value=2.0,  max_value=5.0,  value=3.08, step=0.01, format="%.2f", key="acd_od")
    acw_od = st.number_input("ACW [mm]", min_value=10.0, max_value=14.0, value=11.70, step=0.01, format="%.2f", key="acw_od")

with col_os:
    st.markdown("### 👁 OS (左眼)")
    lv_os  = st.number_input("LV [mm]",  min_value=-1.0, max_value=1.0,  value=0.03, step=0.01, format="%.2f", key="lv_os")
    cct_os = st.number_input("CCT [µm]", min_value=400,  max_value=700,  value=530,  step=1,                   key="cct_os")
    acd_os = st.number_input("ACD [mm]", min_value=2.0,  max_value=5.0,  value=3.08, step=0.01, format="%.2f", key="acd_os")
    acw_os = st.number_input("ACW [mm]", min_value=10.0, max_value=14.0, value=11.70, step=0.01, format="%.2f", key="acw_os")

st.markdown("---")

# ---- 予測実行 ----
if st.button("🔮 予測実行", type="primary", use_container_width=True):

    res_od = run_all_predictions(lv_od, acd_od, cct_od, acw_od, sex, age)
    res_os = run_all_predictions(lv_os, acd_os, cct_os, acw_os, sex, age)

    # セッションに保存（PDFボタン用）
    st.session_state['pred'] = dict(
        sex=sex, age=age,
        inputs_od=dict(lv=lv_od, cct_um=cct_od, acd=acd_od, acw=acw_od),
        pred_od=res_od[0], size_results_od=res_od[3],
        prob_display_od=res_od[1], size_cands_od=res_od[2],
        inputs_os=dict(lv=lv_os, cct_um=cct_os, acd=acd_os, acw=acw_os),
        pred_os=res_os[0], size_results_os=res_os[3],
        prob_display_os=res_os[1], size_cands_os=res_os[2],
    )

    col_r_od, col_r_os = st.columns(2)
    with col_r_od:
        st.subheader("👁 OD (右眼) Results")
        show_eye_results(*res_od)
    with col_r_os:
        st.subheader("👁 OS (左眼) Results")
        show_eye_results(*res_os)

    st.markdown("---")
    st.info(
        "💡 **Note**: Vault that is too high should be avoided — when in doubt, choose a smaller size.\n\n"
        "This app is for reference only. Final size selection should be based on physician judgment."
    )

# ---- PDF ダウンロードボタン ----
if 'pred' in st.session_state:
    p = st.session_state['pred']
    pdf_bytes = generate_pdf(
        p['sex'], p['age'],
        p['inputs_od'], p['pred_od'], p['size_results_od'],
        p['prob_display_od'], p['size_cands_od'],
        p['inputs_os'], p['pred_os'], p['size_results_os'],
        p['prob_display_os'], p['size_cands_os'],
    )
    filename = f"IPCLsize_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )

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
