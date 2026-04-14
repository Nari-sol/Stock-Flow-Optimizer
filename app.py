import streamlit as st
import pandas as pd
import io
import unicodedata

# ページ設定
st.set_page_config(page_title="Stock-Flow Optimizer", layout="wide")

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------

def normalize_sku(text):
    """SKUの表記ゆれ（大文字小文字、全角半角、あらゆる空白、各種ハイフン）を完全に解消する"""
    if pd.isna(text): return ""
    s = str(text)
    # NFKC正規化 (全角数字・英字を半角に変換)
    s = unicodedata.normalize('NFKC', s)
    # 大文字化
    s = s.upper()
    # あらゆる横棒記号を半角ハイフンに統一
    for char in ['ー', '－', '―', '‐', '−']:
        s = s.replace(char, '-')
    # 品番内のすべての空白（全角半角スペース、タブ、改行）を完全に削除
    s = "".join(s.split())
    return s

def clean_num(val):
    """数値文字列から記号を除去して数値化"""
    if pd.isna(val): return 0.0
    s = str(val).replace(',', '').replace('¥', '').replace('円', '').replace(' ', '').strip()
    try:
        return float(s)
    except:
        return 0.0

def categorize_item(name):
    """商品名からカテゴリを自動判定"""
    CATEGORY_MAP = {
        "IGNITION COIL": "イグニッションコイル",
        "BRAKE": "ブレーキ関連",
        "RADIATOR": "ラジエーター関連",
        "ALTERNATOR": "オルタネーター関連"
    }
    n = str(name).upper()
    for kw, label in CATEGORY_MAP.items():
        if kw in n: return label
    return "その他"

# -----------------------------------------------------------------------------
# データ処理の核
# -----------------------------------------------------------------------------

def process_and_aggregate_df(df, sum_cols, first_cols=None):
    """品番を親品番で集約し、数値は合算、属性は保持する"""
    if df is None or df.empty: return None
    
    # 必要列の確保
    if 'SKU' not in df.columns: return None
    for c in sum_cols:
        if c not in df.columns: df[c] = 0.0

    # SKUを極限まで正規化
    df['SKU'] = df['SKU'].apply(normalize_sku)
    df = df[df['SKU'] != ""].copy()
    
    # 枝番（ハイフンとアンダーバーの両方）をカットして親品番にする
    df['SKU'] = df['SKU'].str.split('-').str[0]
    df['SKU'] = df['SKU'].str.split('_').str[0]
    
    # 品番の末尾にある「SET」や「SET2」などを確実に取り除く
    # （\d* をつけることで、SETの後ろの数字ごと消し去ります）
    df['SKU'] = df['SKU'].str.replace(r'SET\d*$', '', regex=True)
    
    # 分割後に再度末尾などの空白をケア（念のため）
    df['SKU'] = df['SKU'].str.strip()
    
    agg_rules = {}
    for col in sum_cols:
        df[col] = df[col].apply(clean_num)
        agg_rules[col] = 'sum'
    if first_cols:
        for col in first_cols:
            if col in df.columns: agg_rules[col] = 'first'
            
    return df.groupby('SKU').agg(agg_rules).reset_index()

def robust_rename(df, mapping):
    """カラム名の揺れを吸収"""
    if df is None: return None
    new_map = {}
    norm_existing = {normalize_sku(c): c for c in df.columns}
    for standard, candidates in mapping.items():
        for cand in candidates:
            norm_cand = normalize_sku(cand)
            if norm_cand in norm_existing:
                new_map[norm_existing[norm_cand]] = standard
                break
    return df.rename(columns=new_map)

@st.cache_data
def load_all_data(master_file, ys_files, rk_file, az_files):
    """すべてのデータを統合する"""
    try:
        def read_csv(f, skip=0):
            if f is None: return None
            for enc in ['utf-8-sig', 'shift-jis', 'cp932']:
                try:
                    f.seek(0)
                    return pd.read_csv(f, encoding=enc, skiprows=skip)
                except: continue
            return None

        # 1. マスター
        m_raw = read_csv(master_file, skip=4)
        if m_raw is None: return None
        m_raw = robust_rename(m_raw, {
            'SKU': ['コード', '商品コード', 'SKU', '商品ID'],
            '商品名': ['商品名', '品名'],
            '現在の在庫数': ['残数量', '在庫数'],
            '在庫評価単価': ['在庫評価単価', '単価', '評価単価']
        })
        m_raw['在庫金額'] = m_raw.get('現在の在庫数', 0).apply(clean_num) * m_raw.get('在庫評価単価', 0).apply(clean_num)
        m_raw['カテゴリ'] = m_raw.get('商品名', '不明').apply(categorize_item)
        df_master = process_and_aggregate_df(m_raw, ['現在の在庫数', '在庫金額'], ['カテゴリ'])

        # 2. 各モール
        mall_data = []
        configs = [
            (ys_files, 'YS_売上', ['商品コード', 'SKU', '個別商品コード', '商品ID'], ['注文点数合計', '販売数', '売上数量', '数量'], 0),
            (rk_file, '楽天_売上', ['商品管理番号', 'SKU', '商品番号'], ['売上個数', '販売数', '販売個数'], 6),
            (az_files, 'Amazon_売上', ['SKU', '商品コード', '出品者SKU'], ['注文された商品点数', '販売数', '数量'], 0)
        ]
        
        for files, col_name, skus, vals, skip in configs:
            f_list = files if isinstance(files, list) else ([files] if files else [])
            dfs = []
            for f in f_list:
                tmp = read_csv(f, skip=skip)
                if tmp is not None:
                    tmp = robust_rename(tmp, {'SKU': skus, '販売数': vals})
                    if 'SKU' in tmp.columns:
                        if '販売数' not in tmp.columns: tmp['販売数'] = 0
                        dfs.append(tmp[['SKU', '販売数']])
            
            if dfs:
                df_c = pd.concat(dfs, ignore_index=True).rename(columns={'販売数': col_name})
                mall_data.append(process_and_aggregate_df(df_c, [col_name]))
            else:
                mall_data.append(None)

        # 結合
        if df_master is None: return None
        res = df_master.copy()
        for m_df in mall_data:
            if m_df is not None:
                res = pd.merge(res, m_df, on='SKU', how='left')
        
        return res.fillna(0)
    except Exception as e:
        st.error(f"データ統合中にエラーが発生しました: {e}")
        return None

# -----------------------------------------------------------------------------
# UI表示
# -----------------------------------------------------------------------------

st.title("📦 Stock-Flow Optimizer")

st.sidebar.header("📂 データ読み込み")
master_f = st.sidebar.file_uploader("1. マスター (CSV)", type="csv")
ys_fs = st.sidebar.file_uploader("2. Yahoo (CSV / 複数可)", type="csv", accept_multiple_files=True)
rk_f = st.sidebar.file_uploader("3. 楽天 (CSV)", type="csv")
az_fs = st.sidebar.file_uploader("4. Amazon (CSV / 複数可)", type="csv", accept_multiple_files=True)

if master_f:
    df = load_all_data(master_f, ys_fs, rk_f, az_fs)
    if df is not None:
        # カテゴリフィルタ
        unique_cats = sorted(df['カテゴリ'].unique())
        sel_cats = st.sidebar.multiselect("🔎 カテゴリ絞り込み", unique_cats)
        if sel_cats: df = df[df['カテゴリ'].isin(sel_cats)]

        st.write("---")
        mode = st.radio("🔍 分析モード", ["滞留在庫の抽出", "需要の偏りを抽出"], horizontal=True, key="mode_radio", index=0)
        
        # 解説 (絶対に維持)
        if mode == "滞留在庫の抽出":
            baseline = st.slider("在庫の基準値", 0, 100, 10)
            st.info(f"**【滞留在庫の抽出モード】**\n* **概要**: 全モールで売上0の商品を特定。\n* **対象基準**: 在庫 **{baseline}個以上**。\n* **活用シーン**: 長期滞留の特定、セール検討。")
            target_df = df[(df['現在の在庫数'] >= baseline) & (df['YS_売上']==0) & (df['楽天_売上']==0) & (df['Amazon_売上']==0)].copy()
            target_df = target_df.sort_values('現在の在庫数', ascending=False)
        else:
            st.info("【需要の偏りを抽出モード】\n・概要: モール間で販売数に大きな差がある商品を特定。\n・対象基準: 在庫1個以上。\n・活用シーン: 在庫移動、広告戦略の最適化。")
            
            # まず全データに対して合計とスコアを計算する
            df['合計売上'] = df['YS_売上'] + df['楽天_売上'] + df['Amazon_売上']
            df['機会損失スコア'] = df[['YS_売上','楽天_売上','Amazon_売上']].max(axis=1) - df[['YS_売上','楽天_売上','Amazon_売上']].min(axis=1)
            
            # スライダーで「どれくらいの差があったら抽出するか」を決められるようにする
            bias_threshold = st.slider("偏りと判定するスコア（最大と最小の差）", 1, 500, 30)
            
            # 0個縛りをなくし、スコアが基準値以上のものを抽出する
            target_df = df[(df['現在の在庫数'] >= 1) & (df['合計売上'] >= 1) & (df['機会損失スコア'] >= bias_threshold)].copy()
            
            target_df['合計売上'] = target_df['合計売上'].fillna(0).astype(int)
            target_df['機会損失スコア'] = target_df['機会損失スコア'].fillna(0).astype(int)

            # 各行の売上状況テキストを「0かどうか」ではなく「順位」で判定するように進化
            def generate_status_text(row):
                sales_dict = {'Yahoo': row['YS_売上'], '楽天': row['楽天_売上'], 'Amazon': row['Amazon_売上']}
                # 売上が高い順に並び替え
                sorted_sales = sorted(sales_dict.items(), key=lambda item: item[1], reverse=True)
                # 1位を主力、3位を課題として表示
                return f"主力:{sorted_sales[0][0]} / 課題:{sorted_sales[2][0]}"

            target_df['偏り状況'] = target_df.apply(generate_status_text, axis=1)
            
            # スコア順に並び替え
            target_df = target_df.sort_values('機会損失スコア', ascending=False)

        # サマリー
        st.subheader("📊 分析サマリー")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("抽出品番数", f"{len(target_df)} 件")
        c2.metric("総在庫数", f"{int(target_df['現在の在庫数'].sum()):,}")
        c3.metric("総在庫金額", f"¥ {int(target_df['在庫金額'].sum()):,}")
        c4.metric("対象品合計売上", f"{int(target_df[['YS_売上','楽天_売上','Amazon_売上']].sum().sum()):,}")

        # スタイリング表示
        def style_df(d):
            cols = ['YS_売上','楽天_売上','Amazon_売上','現在の在庫数','在庫金額', '合計売上', '機会損失スコア', '偏り状況']
            s_cols = ['YS_売上','楽天_売上','Amazon_売上']
            
            # 安全な型変換
            tmp_d = d.copy()
            for c in cols:
                if c in tmp_d.columns:
                    # 数値列のみ整数に変換するよう、列名を指定して処理を分けます
                    if c in ['YS_売上','楽天_売上','Amazon_売上','現在の在庫数','在庫金額', '合計売上', '機会損失スコア']:
                        tmp_d[c] = pd.to_numeric(tmp_d[c], errors='coerce').fillna(0).astype(int)
            
            return tmp_d.style.format({c: "{:,}" for c in cols if c in tmp_d.columns and c != '偏り状況'}).map(
                lambda v: 'color: #FF4B4B; font-weight: bold;' if v == 0 else ('color: #0099FF;' if v > 0 else ''),
                subset=[c for c in s_cols if c in tmp_d.columns]
            )
            

        if not target_df.empty:
            st.dataframe(style_df(target_df), use_container_width=True)
            csv = target_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📄 CSVダウンロード", csv, f"result_{pd.Timestamp.now():%Y%m%d}.csv", "text/csv")
        else:
            st.info("対象データはありません。")
else:
    st.info("👈 サイドバーからCSVをアップロードしてください。")
