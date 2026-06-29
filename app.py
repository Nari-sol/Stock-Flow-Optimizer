import streamlit as st
import pandas as pd
import io
import unicodedata
import re

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

def normalize_month(sheet_name):
    """シート名から数値部分を抽出して『X月』の形式に統一する。数値がない場合はそのままトリムして返す"""
    if pd.isna(sheet_name): return "不明"
    s = str(sheet_name).strip()
    match = re.search(r'(\d+)', s)
    if match:
        return f"{int(match.group(1))}月"
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

def process_and_aggregate_df(df, sum_cols, first_cols=None, group_cols=['SKU']):
    """品番（および指定列）で集約し、数値は合算、属性は保持する"""
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
            
    return df.groupby(group_cols).agg(agg_rules).reset_index()

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
def load_all_data(master_file, ys_files, rk_files, az_files):
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
        m_raw['在庫評価単価'] = m_raw.get('在庫評価単価', 0).apply(clean_num)
        m_raw['在庫金額'] = m_raw.get('現在の在庫数', 0).apply(clean_num) * m_raw['在庫評価単価']
        m_raw['カテゴリ'] = m_raw.get('商品名', '不明').apply(categorize_item)
        df_master = process_and_aggregate_df(m_raw, ['現在の在庫数', '在庫金額'], ['カテゴリ', '在庫評価単価'])

        # 2. 各モール
        # 各モールデータ読み込み（シートごとに月を付与）
        ys_dfs = []
        ys_f_list = ys_files if isinstance(ys_files, list) else ([ys_files] if ys_files else [])
        for ys_f in ys_f_list:
            if ys_f is not None:
                try:
                    ys_f.seek(0)
                    excel_sheets = pd.read_excel(ys_f, sheet_name=None)
                    for sheet_name, sheet_df in excel_sheets.items():
                        if sheet_df is not None and not sheet_df.empty:
                            sheet_df = robust_rename(sheet_df, {
                                'SKU': ['商品コード', 'SKU', '個別商品コード', '商品ID'],
                                '販売数': ['注文点数合計', '販売数', '売上数量', '数量']
                            })
                            if 'SKU' in sheet_df.columns:
                                if '販売数' not in sheet_df.columns: 
                                    sheet_df['販売数'] = 0
                                sheet_df['月'] = normalize_month(sheet_name)
                                ys_dfs.append(sheet_df[['SKU', '販売数', '月']])
                except Exception as e:
                    st.error(f"Yahoo Excelファイル ({ys_f.name}) の読み込み中にエラーが発生しました: {e}")

        rk_dfs = []
        rk_f_list = rk_files if isinstance(rk_files, list) else ([rk_files] if rk_files else [])
        for rk_f in rk_f_list:
            if rk_f is not None:
                try:
                    rk_f.seek(0)
                    excel_sheets = pd.read_excel(rk_f, sheet_name=None)
                    for sheet_name, sheet_df in excel_sheets.items():
                        if sheet_df is not None and not sheet_df.empty:
                            sheet_df = robust_rename(sheet_df, {
                                'SKU': ['商品管理番号', 'SKU', '商品番号'],
                                '販売数': ['売上個数', '販売数', '販売個数']
                            })
                            if 'SKU' in sheet_df.columns:
                                if '販売数' not in sheet_df.columns: 
                                    sheet_df['販売数'] = 0
                                sheet_df['月'] = normalize_month(sheet_name)
                                rk_dfs.append(sheet_df[['SKU', '販売数', '月']])
                except Exception as e:
                    st.error(f"楽天 Excelファイル ({rk_f.name}) の読み込み中にエラーが発生しました: {e}")

        az_dfs = []
        az_f_list = az_files if isinstance(az_files, list) else ([az_files] if az_files else [])
        for az_f in az_f_list:
            if az_f is not None:
                try:
                    az_f.seek(0)
                    excel_sheets = pd.read_excel(az_f, sheet_name=None)
                    for sheet_name, sheet_df in excel_sheets.items():
                        if sheet_df is not None and not sheet_df.empty:
                            sheet_df = robust_rename(sheet_df, {
                                'SKU': ['SKU', '商品コード', '出品者SKU'], 
                                '販売数': ['注文された商品点数', '販売数', '数量']
                            })
                            if 'SKU' in sheet_df.columns:
                                if '販売数' not in sheet_df.columns: 
                                    sheet_df['販売数'] = 0
                                sheet_df['月'] = normalize_month(sheet_name)
                                az_dfs.append(sheet_df[['SKU', '販売数', '月']])
                except Exception as e:
                    st.error(f"Amazon Excelファイル ({az_f.name}) の読み込み中にエラーが発生しました: {e}")

        # すべての存在する「月」をユニークに抽出
        all_months = set()
        for df_list in [ys_dfs, rk_dfs, az_dfs]:
            for tmp in df_list:
                if '月' in tmp.columns:
                    all_months.update(tmp['月'].dropna().unique())
        
        if not all_months:
            all_months = {'不明'}
            
        all_months = sorted(list(all_months))

        # マスターの全SKUとall_monthsのデカルト積（ベースフレーム）を作成
        sku_month_base = []
        if df_master is not None and not df_master.empty:
            for sku in df_master['SKU'].unique():
                for m in all_months:
                    sku_month_base.append({'SKU': sku, '月': m})
        df_base = pd.DataFrame(sku_month_base)

        if df_base.empty:
            return None

        # 各モールデータを ['SKU', '月'] で集約
        mall_data = []
        
        # Yahoo
        if ys_dfs:
            ys_concat = pd.concat(ys_dfs, ignore_index=True).rename(columns={'販売数': 'YS_売上'})
            ys_agg = process_and_aggregate_df(ys_concat, ['YS_売上'], group_cols=['SKU', '月'])
            mall_data.append((ys_agg, 'YS_売上'))
        else:
            mall_data.append((None, 'YS_売上'))

        # 楽天
        if rk_dfs:
            rk_concat = pd.concat(rk_dfs, ignore_index=True).rename(columns={'販売数': '楽天_売上'})
            rk_agg = process_and_aggregate_df(rk_concat, ['楽天_売上'], group_cols=['SKU', '月'])
            mall_data.append((rk_agg, '楽天_売上'))
        else:
            mall_data.append((None, '楽天_売上'))

        # Amazon
        if az_dfs:
            az_concat = pd.concat(az_dfs, ignore_index=True).rename(columns={'販売数': 'Amazon_売上'})
            az_agg = process_and_aggregate_df(az_concat, ['Amazon_売上'], group_cols=['SKU', '月'])
            mall_data.append((az_agg, 'Amazon_売上'))
        else:
            mall_data.append((None, 'Amazon_売上'))

        # 結合
        res = pd.merge(df_base, df_master, on='SKU', how='left')
        for m_df, col in mall_data:
            if m_df is not None:
                res = pd.merge(res, m_df, on=['SKU', '月'], how='left')
            else:
                res[col] = 0.0

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
ys_fs = st.sidebar.file_uploader("2. Yahoo (Excel / 複数可)", type=["xlsx", "xls"], accept_multiple_files=True)
rk_fs = st.sidebar.file_uploader("3. 楽天 (Excel / 複数可)", type=["xlsx", "xls"], accept_multiple_files=True)
az_fs = st.sidebar.file_uploader("4. Amazon (Excel / 複数可)", type=["xlsx", "xls"], accept_multiple_files=True)

if master_f:
    df = load_all_data(master_f, ys_fs, rk_fs, az_fs)
    if df is not None:
        # カテゴリフィルタ
        unique_cats = sorted(df['カテゴリ'].unique())
        sel_cats = st.sidebar.multiselect("🔎 カテゴリ絞り込み", unique_cats)
        if sel_cats: df = df[df['カテゴリ'].isin(sel_cats)]

        st.write("---")
        mode = st.radio("🔍 分析モード", ["滞留在庫の抽出", "需要の偏りを抽出", "月別効果検証（比較）"], horizontal=True, key="mode_radio", index=0)
        
        unique_months = sorted(df['月'].unique())

        # スタイリング関数の定義
        def style_df(d):
            cols = ['現在の在庫数', '在庫金額', '偏り状況']
            s_cols = []
            
            if mode in ["滞留在庫の抽出", "需要の偏りを抽出"]:
                cols += ['YS_売上','楽天_売上','Amazon_売上', '合計売上', '機会損失スコア']
                s_cols += ['YS_売上','楽天_売上','Amazon_売上']
            else:
                cols += [
                    '比較元_YS_売上', '比較元_楽天_売上', '比較元_Amazon_売上', '比較元_合計売上',
                    '比較元_合計売上金額',
                    '比較先_YS_売上', '比較先_楽天_売上', '比較先_Amazon_売上', '比較先_合計売上',
                    '比較先_合計売上金額',
                    '売上個数変化量', '売上金額変化量'
                ]
                s_cols += [
                    '比較元_YS_売上', '比較元_楽天_売上', '比較元_Amazon_売上',
                    '比較先_YS_売上', '比較先_楽天_売上', '比較先_Amazon_売上'
                ]
            
            # 安全な型変換
            tmp_d = d.copy()
            for c in cols:
                if c in tmp_d.columns and c != '偏り状況':
                    tmp_d[c] = pd.to_numeric(tmp_d[c], errors='coerce').fillna(0).astype(int)
            
            style_obj = tmp_d.style.format({c: "{:,}" for c in cols if c in tmp_d.columns and c != '偏り状況'})
            
            # 配色ルール: 売上0=赤, 売上あり=青
            # 変化量: プラス=青, マイナス=赤
            def get_color_map(val, is_diff=False):
                if is_diff:
                    if val > 0: return 'color: #0099FF; font-weight: bold;'
                    elif val < 0: return 'color: #FF4B4B; font-weight: bold;'
                    return ''
                else:
                    if val == 0: return 'color: #FF4B4B; font-weight: bold;'
                    elif val > 0: return 'color: #0099FF;'
                    return ''
            
            for c in s_cols:
                if c in tmp_d.columns:
                    style_obj = style_obj.map(lambda v: get_color_map(v), subset=[c])
            
            if '売上個数変化量' in tmp_d.columns:
                style_obj = style_obj.map(lambda v: get_color_map(v, is_diff=True), subset=['売上個数変化量'])
            if '売上金額変化量' in tmp_d.columns:
                style_obj = style_obj.map(lambda v: get_color_map(v, is_diff=True), subset=['売上金額変化量'])
            
            return style_obj

        if mode in ["滞留在庫の抽出", "需要の偏りを抽出"]:
            # 通常モード: 表示対象の月を1つ選択
            selected_month = st.selectbox("📅 表示対象の月を選択", unique_months, index=0)
            df_active = df[df['月'] == selected_month].copy()

            if mode == "滞留在庫の抽出":
                baseline = st.slider("在庫の基準値", 0, 100, 10)
                st.info(f"**【滞留在庫の抽出モード】**\n* **対象月**: {selected_month}\n* **概要**: 全モールで売上0の商品を特定。\n* **対象基準**: 在庫 **{baseline}個以上**。\n* **活用シーン**: 長期滞留の特定、セール検討。")
                target_df = df_active[(df_active['現在の在庫数'] >= baseline) & (df_active['YS_売上']==0) & (df_active['楽天_売上']==0) & (df_active['Amazon_売上']==0)].copy()
                target_df = target_df.sort_values('現在の在庫数', ascending=False)
            else:
                st.info(f"**【需要の偏りを抽出モード】**\n* **対象月**: {selected_month}\n* **概要**: モール間で販売数に大きな差がある商品を特定。\n* **対象基準**: 在庫1個以上。\n* **活用シーン**: 在庫移動、広告戦略の最適化。")
                
                # まず全データに対して合計とスコアを計算する
                df_active['合計売上'] = df_active['YS_売上'] + df_active['楽天_売上'] + df_active['Amazon_売上']
                df_active['機会損失スコア'] = df_active[['YS_売上','楽天_売上','Amazon_売上']].max(axis=1) - df_active[['YS_売上','楽天_売上','Amazon_売上']].min(axis=1)
                
                # スライダーで「どれくらいの差があったら抽出するか」を決められるようにする
                bias_threshold = st.slider("偏りと判定するスコア（最大と最小の差）", 1, 500, 30)
                
                # 0個縛りをなくし、スコアが基準値以上のものを抽出する
                target_df = df_active[(df_active['現在の在庫数'] >= 1) & (df_active['合計売上'] >= 1) & (df_active['機会損失スコア'] >= bias_threshold)].copy()
                
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

            # 通常モードのサマリー
            st.subheader("📊 分析サマリー")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("抽出品番数", f"{len(target_df)} 件")
            c2.metric("総在庫数", f"{int(target_df['現在の在庫数'].sum()):,}")
            c3.metric("総在庫金額", f"¥ {int(target_df['在庫金額'].sum()):,}")
            c4.metric("対象品合計売上", f"{int(target_df[['YS_売上','楽天_売上','Amazon_売上']].sum().sum()):,}")

            # テーブル表示
            if not target_df.empty:
                st.dataframe(style_df(target_df), use_container_width=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    target_df.to_excel(writer, index=False)
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📄 Excelダウンロード",
                    data=excel_data,
                    file_name=f"result_{pd.Timestamp.now():%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("対象データはありません。")

        else: # 月別効果検証（比較）
            st.info("**【月別効果検証モード】**\n* **概要**: 選択した複数月の期間における売上の変化（個数・金額）を品番ごとに比較検証します。")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                months_src = st.multiselect("📅 比較元（施策前）の月", unique_months, default=None)
            with col_m2:
                months_dst = st.multiselect("📅 比較先（施策後）の月", unique_months, default=None)

            if not months_src or not months_dst:
                st.warning("⚠️ 比較元（施策前）と比較先（施策後）の月をそれぞれ1つ以上選択してください。")
            else:
                # 比較元のデータを抽出して集計
                df_src = df[df['月'].isin(months_src)][['SKU', 'YS_売上', '楽天_売上', 'Amazon_売上']].copy()
                df_src = df_src.groupby('SKU').sum().reset_index()
                df_src = df_src.rename(columns={
                    'YS_売上': '比較元_YS_売上',
                    '楽天_売上': '比較元_楽天_売上',
                    'Amazon_売上': '比較元_Amazon_売上'
                })
                
                # 比較先のデータを抽出して集計
                df_dst = df[df['月'].isin(months_dst)][['SKU', 'YS_売上', '楽天_売上', 'Amazon_売上']].copy()
                df_dst = df_dst.groupby('SKU').sum().reset_index()
                df_dst = df_dst.rename(columns={
                    'YS_売上': '比較先_YS_売上',
                    '楽天_売上': '比較先_楽天_売上',
                    'Amazon_売上': '比較先_Amazon_売上'
                })
                
                # マスター情報（月情報を含まないユニークなもの）
                df_meta = df[['SKU', 'カテゴリ', '現在の在庫数', '在庫金額', '在庫評価単価']].drop_duplicates(subset=['SKU']).copy()
                
                # マージ
                target_df = pd.merge(df_meta, df_src, on='SKU', how='left')
                target_df = pd.merge(target_df, df_dst, on='SKU', how='left')
                target_df = target_df.fillna(0)
                
                # 合計売上の計算
                target_df['比較元_合計売上'] = target_df['比較元_YS_売上'] + target_df['比較元_楽天_売上'] + target_df['比較元_Amazon_売上']
                target_df['比較先_合計売上'] = target_df['比較先_YS_売上'] + target_df['比較先_楽天_売上'] + target_df['比較先_Amazon_売上']
                
                # 合計売上金額の計算
                target_df['比較元_合計売上金額'] = target_df['比較元_合計売上'] * target_df['在庫評価単価']
                target_df['比較先_合計売上金額'] = target_df['比較先_合計売上'] * target_df['在庫評価単価']
                
                # 売上個数変化量・売上金額変化量の計算
                target_df['売上個数変化量'] = target_df['比較先_合計売上'] - target_df['比較元_合計売上']
                target_df['売上金額変化量'] = target_df['比較先_合計売上金額'] - target_df['比較元_合計売上金額']
                
                # 変化量順にソート (個数の変化量を基準)
                target_df = target_df.sort_values('売上個数変化量', ascending=False)

                # 月別効果検証用のサマリー
                st.subheader("📊 分析サマリー")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("抽出品番数", f"{len(target_df)} 件")
                c2.metric("総在庫数", f"{int(target_df['現在の在庫数'].sum()):,}")
                c3.metric("総在庫金額", f"¥ {int(target_df['在庫金額'].sum()):,}")
                c4.metric("総売上個数変化量", f"{int(target_df['売上個数変化量'].sum()):+}")
                c5.metric("総売上金額変化量", f"¥ {int(target_df['売上金額変化量'].sum()):+}")

                # テーブル表示
                if not target_df.empty:
                    st.dataframe(style_df(target_df), use_container_width=True)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        target_df.to_excel(writer, index=False)
                    excel_data = buffer.getvalue()
                    st.download_button(
                        label="📄 Excelダウンロード",
                        data=excel_data,
                        file_name=f"result_{pd.Timestamp.now():%Y%m%d}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("対象データはありません。")
else:
    st.info("👈 サイドバーからCSVをアップロードしてください。")
