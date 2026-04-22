import io
import os
import base64
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import missingno as msno
from wordcloud import WordCloud
from scipy.stats import pearsonr, kruskal, chi2_contingency

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Script en: UA_MDM_Labo2/input/Notebook generados/EDA/
# BASE apunta a: UA_MDM_Labo2/input/
BASE   = Path(__file__).resolve().parent.parent.parent
OUTPUT = Path(__file__).resolve().parent / 'output'
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Constantes ────────────────────────────────────────────────────────────────
ADOPTION_LABELS = {
    0: 'Mismo día',
    1: '1-7 días',
    2: '8-30 días',
    3: '31-90 días',
    4: '>100 días',
}
ADOPTION_ORDER = [ADOPTION_LABELS[i] for i in range(5)]

C_ORANGE = '#f7931a'
C_BLUE   = '#3b82f6'
C_GREEN  = '#10b981'
C_PURPLE = '#8b5cf6'
C_RED    = '#ef4444'
C_TEAL   = '#14b8a6'
C_AMBER  = '#f59e0b'
C_INDIGO = '#6366f1'
PALETTE  = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_RED, C_TEAL, C_AMBER, C_INDIGO]

SIDEBAR_BG   = '#1a1f36'
SIDEBAR_W    = '240px'
CONTENT_BG   = '#f8fafc'
CARD_BG      = '#ffffff'
TEXT_PRIMARY = '#1a1f36'
TEXT_MUTED   = '#6b7280'
BORDER_LIGHT = '#e5e7eb'


# ── Utilidades ────────────────────────────────────────────────────────────────
def hex_rgba(hex_c, alpha=0.15):
    h = hex_c.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def save_mpl(fig, name):
    try:
        fig.savefig(OUTPUT / f'{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    except Exception:
        pass


def mpl_to_b64(fig, name=None):
    if name:
        save_mpl(fig, name)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{enc}'


def chart_layout(title='', xt='', yt='', height=360, showlegend=True):
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY,
                                          family='Inter, Roboto, Arial'), x=0, xanchor='left'),
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Roboto, Arial, sans-serif', color=TEXT_MUTED, size=11),
        margin=dict(l=50, r=20, t=55, b=65),
        xaxis=dict(title=xt, showgrid=False, zeroline=False,
                   tickfont=dict(size=10), title_font=dict(size=11)),
        yaxis=dict(title=yt, gridcolor='#f0f4f8', zeroline=False,
                   tickfont=dict(size=10), title_font=dict(size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=10)),
        showlegend=showlegend,
        height=height,
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, Arial'),
    )


def top_n(series, n=9):
    counts = series.value_counts()
    if len(counts) <= n:
        return counts
    return pd.concat([counts.iloc[:n], pd.Series({'Otros': counts.iloc[n:].sum()})])


def px_bar(x_vals, y_vals, title, xt='', yt='Cantidad', color=None,
           horizontal=False, height=360):
    c = color or C_BLUE
    if horizontal:
        fig = go.Figure(go.Bar(x=y_vals, y=x_vals, orientation='h',
                                marker_color=c, marker_line_width=0, opacity=0.88))
        fig.update_layout(**chart_layout(title, xt=yt, yt='', height=height, showlegend=False))
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(gridcolor='#f0f4f8')
    else:
        fig = go.Figure(go.Bar(x=x_vals, y=y_vals,
                                marker_color=c, marker_line_width=0, opacity=0.88))
        fig.update_layout(**chart_layout(title, xt=xt, yt=yt, height=height, showlegend=False))
    return fig


def px_area_bar(x_vals, y_vals, title, xt='', yt='', color=None, height=380):
    c = color or C_BLUE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines+markers',
        line=dict(color=c, width=2.5, shape='spline', smoothing=1.2),
        fill='tozeroy', fillcolor=hex_rgba(c, 0.15),
        marker=dict(size=7, color=c, line=dict(color='white', width=2)),
    ))
    fig.update_layout(**chart_layout(title, xt=xt, yt=yt, height=height, showlegend=False))
    return fig


def px_box_chart(series, title, yt='', color=None, height=360):
    c = color or C_BLUE
    clean = series.dropna()
    fig = go.Figure(go.Box(
        y=clean, name='',
        marker_color=c, line_color=c,
        fillcolor=hex_rgba(c, 0.2),
        boxmean='sd',
        jitter=0.1, pointpos=0,
        marker=dict(size=3, opacity=0.3),
    ))
    fig.update_layout(**chart_layout(title, yt=yt, height=height, showlegend=False))
    fig.update_xaxes(showticklabels=False)
    return fig


def px_heatmap_corr(corr_m):
    fig = px.imshow(
        corr_m, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        aspect='auto', text_auto='.2f',
    )
    fig.update_traces(textfont=dict(size=8))
    fig.update_layout(
        **chart_layout('Matriz de correlación — variables numéricas',
                        height=max(420, len(corr_m) * 30)),
        coloraxis_colorbar=dict(thickness=14, len=0.7),
    )
    return fig


def px_grouped_bar(x_vals, y1, y2, name1, name2, title, xt='', yt='', height=400):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_vals, y=y1, name=name1,
                          marker_color=C_BLUE, marker_line_width=0, opacity=0.88))
    fig.add_trace(go.Bar(x=x_vals, y=y2, name=name2,
                          marker_color=C_ORANGE, marker_line_width=0, opacity=0.88))
    fig.update_layout(**chart_layout(title, xt=xt, yt=yt, height=height), barmode='group')
    return fig


def px_multibar_stacked(cross_df, title, xt='', yt='', height=400):
    fig = go.Figure()
    for idx, col_name in enumerate(cross_df.index):
        fig.add_trace(go.Bar(
            x=cross_df.columns.tolist(),
            y=cross_df.loc[col_name].values,
            name=str(col_name),
            marker_color=PALETTE[idx % len(PALETTE)],
            marker_line_width=0, opacity=0.88,
        ))
    fig.update_layout(**chart_layout(title, xt=xt, yt=yt, height=height), barmode='group')
    return fig


def graph(fig, fname=None):
    if fname:
        try:
            fig.write_image(str(OUTPUT / f'{fname}.png'), scale=2)
        except Exception:
            pass
    return dcc.Graph(figure=fig, config={'displayModeBar': 'hover',
                                          'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def mpl_img(fig, name=None):
    return html.Img(src=mpl_to_b64(fig, name), style={'width': '100%', 'display': 'block'})


def card(children, style_extra=None):
    base = {'background': CARD_BG, 'borderRadius': '16px', 'border': 'none',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.07)', 'padding': '1.25rem',
            'marginBottom': '1.25rem'}
    if style_extra:
        base.update(style_extra)
    return html.Div(children, style=base)


def kpi_card(label, value, subtitle, icon, color):
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(icon, style={
                        'width': '46px', 'height': '46px', 'borderRadius': '12px',
                        'background': hex_rgba(color, 0.15),
                        'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'center', 'fontSize': '1.3rem',
                    })
                ], width='auto', className='pe-0'),
                dbc.Col([
                    html.P(label, style={'color': TEXT_MUTED, 'fontSize': '0.78rem',
                                         'fontWeight': '500', 'margin': '0',
                                         'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                    html.H4(str(value), style={'color': TEXT_PRIMARY, 'fontSize': '1.7rem',
                                               'fontWeight': '700', 'margin': '0 0 2px'}),
                    html.Small(subtitle, style={'color': '#9ca3af', 'fontSize': '0.72rem'}),
                ]),
            ], align='center', className='g-2'),
        ], className='py-3'),
    ], style={'border': 'none', 'borderRadius': '16px',
               'boxShadow': '0 1px 4px rgba(0,0,0,0.08)',
               'borderTop': f'3px solid {color}', 'background': CARD_BG})


def section_title(text):
    return html.H5(text, style={'color': TEXT_PRIMARY, 'fontWeight': '600',
                                 'marginBottom': '1rem', 'marginTop': '0.5rem',
                                 'borderLeft': f'4px solid {C_BLUE}',
                                 'paddingLeft': '0.75rem'})


def sub_title(text):
    return html.H6(text, style={'color': TEXT_MUTED, 'fontWeight': '600',
                                 'marginBottom': '0.75rem', 'fontSize': '0.85rem',
                                 'textTransform': 'uppercase', 'letterSpacing': '0.5px'})


def divider():
    return html.Hr(style={'borderColor': BORDER_LIGHT, 'margin': '1.5rem 0'})


def dtable(df_t, page_size=15):
    return dash_table.DataTable(
        data=df_t.astype(str).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df_t.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'fontFamily': 'Inter, Arial, sans-serif', 'fontSize': 12,
                    'textAlign': 'left', 'padding': '8px 12px', 'border': 'none',
                    'color': TEXT_MUTED},
        style_header={'backgroundColor': '#f0f4f8', 'fontWeight': '600',
                      'color': TEXT_PRIMARY, 'border': 'none', 'fontSize': 11},
        style_data={'borderBottom': f'1px solid {BORDER_LIGHT}'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafcff'}],
        page_size=page_size,
        sort_action='native',
        filter_action='native',
    )


def info_panel(title, items):
    rows = []
    for label, value in items:
        rows.append(html.Div([
            html.Span(label, style={'color': TEXT_MUTED, 'fontSize': '0.8rem'}),
            html.Span(str(value), style={'color': TEXT_PRIMARY, 'fontWeight': '600',
                                          'fontSize': '0.85rem', 'float': 'right'}),
        ], style={'marginBottom': '0.6rem', 'overflow': 'hidden'}))
        rows.append(html.Hr(style={'margin': '0.35rem 0', 'borderColor': '#f0f4f8'}))
    return dbc.Card([
        dbc.CardHeader(html.H6(title, className='mb-0',
                                style={'fontWeight': '700', 'color': TEXT_PRIMARY,
                                       'fontSize': '0.85rem'}),
                       style={'background': 'transparent', 'border': 'none', 'paddingBottom': '0'}),
        dbc.CardBody(rows[:-1], style={'paddingTop': '0.75rem'}),
    ], style={'border': 'none', 'borderRadius': '16px',
               'boxShadow': '0 1px 4px rgba(0,0,0,0.07)', 'background': CARD_BG})


# ── Carga de datos ────────────────────────────────────────────────────────────
def load_main():
    breeds = pd.read_csv(BASE / 'breed_labels.csv')
    colors = pd.read_csv(BASE / 'color_labels.csv')
    states = pd.read_csv(BASE / 'state_labels.csv')
    bmap   = {**dict(zip(breeds['BreedID'], breeds['BreedName'])), 0: 'Desconocido'}
    cmap   = {**dict(zip(colors['ColorID'], colors['ColorName'])), 0: 'Ninguno'}
    smap   = dict(zip(states['StateID'], states['StateName']))
    df     = pd.read_csv(BASE / 'train' / 'train.csv')
    df['Type']         = df['Type'].map({1: 'Perro', 2: 'Gato'})
    df['Gender']       = df['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    df['Breed1']       = df['Breed1'].map(bmap).fillna('Desconocido')
    df['Breed2']       = df['Breed2'].map(bmap).fillna('Desconocido')
    df['State']        = df['State'].map(smap)
    df['MaturitySize'] = df['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'X-Grande'})
    df['FurLength']    = df['FurLength'].map({1: 'Corto', 2: 'Mediano', 3: 'Largo'})
    df['Vaccinated']   = df['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Dewormed']     = df['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Sterilized']   = df['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Health']       = df['Health'].map({1: 'Sano', 2: 'Lesión leve', 3: 'Lesión grave'})
    df['Color_1']      = df['Color1'].map(cmap)
    df['Color_2']      = df['Color2'].map(cmap)
    df['Color_3']      = df['Color3'].map(cmap)
    df = df.drop(columns=['Name', 'Description', 'Color1', 'Color2', 'Color3', 'RescuerID'],
                 errors='ignore')
    return df


def load_metadata():
    meta_dir = BASE / 'train_metadata'
    if not meta_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    records, all_labels = [], []
    for fp in sorted(meta_dir.glob('*.json')):
        pet_id = fp.stem.rsplit('-', 1)[0]
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        lbls = data.get('labelAnnotations', [])
        for lb in lbls:
            all_labels.append({'PetID': pet_id, 'description': lb.get('description', ''),
                                'score': lb.get('score', 0), 'topicality': lb.get('topicality', 0)})
        dr = dg = db = mpf = np.nan
        try:
            clrs = data['imagePropertiesAnnotation']['dominantColors']['colors']
            pfs  = [c.get('pixelFraction', 0) for c in clrs]
            tot  = sum(pfs)
            if tot > 0:
                dr  = sum(c['color'].get('red',   0) * p for c, p in zip(clrs, pfs)) / tot
                dg  = sum(c['color'].get('green', 0) * p for c, p in zip(clrs, pfs)) / tot
                db  = sum(c['color'].get('blue',  0) * p for c, p in zip(clrs, pfs)) / tot
                mpf = max(pfs)
        except (KeyError, TypeError):
            pass
        cc = np.nan
        try:
            hints = data['cropHintsAnnotation']['cropHints']
            if hints:
                cc = float(np.mean([h.get('confidence', 0) for h in hints]))
        except (KeyError, TypeError):
            pass
        records.append({'PetID': pet_id, 'n_labels': len(lbls),
                         'avg_label_score': float(np.mean([lb.get('score', 0) for lb in lbls])) if lbls else np.nan,
                         'dom_R': dr, 'dom_G': dg, 'dom_B': db,
                         'max_pixelFraction': mpf, 'crop_confidence': cc,
                         'has_face': 1 if data.get('faceAnnotations') else 0,
                         'has_text': 1 if data.get('textAnnotations') else 0})
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    mdf = pd.DataFrame(records)
    agg = mdf.groupby('PetID').agg(
        n_labels_img=('n_labels', 'mean'), avg_label_score=('avg_label_score', 'mean'),
        dom_R=('dom_R', 'mean'), dom_G=('dom_G', 'mean'), dom_B=('dom_B', 'mean'),
        max_pixelFraction=('max_pixelFraction', 'mean'), crop_confidence=('crop_confidence', 'mean'),
        has_face=('has_face', 'max'), has_text=('has_text', 'max'),
    ).reset_index()
    return agg, pd.DataFrame(all_labels)


def load_sentiment():
    sent_dir = BASE / 'train_sentiment'
    if not sent_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    records, all_entities = [], []
    for fp in sent_dir.glob('*.json'):
        pet_id = fp.stem
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            continue
        ds    = data.get('documentSentiment', {})
        score = ds.get('score', np.nan)
        mag   = ds.get('magnitude', np.nan)
        sents = data.get('sentences', [])
        s_range = np.nan
        if len(sents) > 1:
            ss = [s.get('sentiment', {}).get('score', 0) for s in sents]
            s_range = max(ss) - min(ss)
        for ent in data.get('entities', []):
            all_entities.append({'PetID': pet_id, 'name': ent.get('name', ''),
                                  'type': ent.get('type', 'OTHER'), 'salience': ent.get('salience', 0)})
        try:
            sc = 'Positiva' if score > 0.25 else ('Negativa' if score < -0.25 else 'Neutra')
        except TypeError:
            sc = 'Neutra'
        records.append({'PetID': pet_id, 'doc_score': score, 'doc_magnitude': mag,
                         'n_sentences': len(sents), 'sentence_score_range': s_range,
                         'sentiment_class': sc})
    return pd.DataFrame(records), pd.DataFrame(all_entities)


def compute_associations(df_in):
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_in.select_dtypes(include='object').columns if c != 'PetID']
    pairs    = []
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            valid = df_in[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            r = valid[c1].corr(valid[c2])
            if not np.isnan(r):
                pairs.append({'var1': c1, 'var2': c2, 'tipo': 'num-num',
                               'medida': abs(float(r)), 'medida_raw': float(r)})
    for cat in cat_cols:
        if df_in[cat].nunique() > 30:
            continue
        for num in num_cols:
            valid = df_in[[cat, num]].dropna()
            if len(valid) < 10 or valid[cat].nunique() < 2:
                continue
            groups     = [valid[num][valid[cat] == c].values for c in valid[cat].unique()]
            grand_mean = valid[num].mean()
            ss_b = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_t = ((valid[num] - grand_mean) ** 2).sum()
            eta  = float(np.sqrt(ss_b / ss_t)) if ss_t > 0 else np.nan
            if not np.isnan(eta):
                pairs.append({'var1': cat, 'var2': num, 'tipo': 'cat-num',
                               'medida': eta, 'medida_raw': eta})
    for i, c1 in enumerate(cat_cols):
        if df_in[c1].nunique() > 30:
            continue
        for c2 in cat_cols[i + 1:]:
            if df_in[c2].nunique() > 30:
                continue
            valid = df_in[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            try:
                ct = pd.crosstab(valid[c1], valid[c2])
                chi2_val, _, _, _ = chi2_contingency(ct)
                n = int(ct.values.sum()); min_dim = min(ct.shape) - 1
                if min_dim > 0 and n > 0:
                    cv = float(np.sqrt(chi2_val / (n * min_dim)))
                    if not np.isnan(cv):
                        pairs.append({'var1': c1, 'var2': c2, 'tipo': 'cat-cat',
                                       'medida': cv, 'medida_raw': cv})
            except Exception:
                pass
    return pd.DataFrame(pairs).sort_values('medida', ascending=False).reset_index(drop=True)


# ── Carga ─────────────────────────────────────────────────────────────────────
print('Cargando datos…')
df_main             = load_main()
print('Cargando metadata de imágenes…')
meta_agg, labels_df = load_metadata()
print('Cargando sentimientos…')
sent_df, ent_df     = load_sentiment()

df = df_main.copy()
if not meta_agg.empty:
    df = df.merge(meta_agg, on='PetID', how='left')
if not sent_df.empty:
    df = df.merge(sent_df,  on='PetID', how='left')

HAS_META = 'avg_label_score' in df.columns
HAS_SENT = 'doc_score'       in df.columns

print('Construyendo dashboard…')

pct_30d   = round(df['AdoptionSpeed'].isin([0, 1, 2]).mean() * 100, 1)
avg_speed = round(df['AdoptionSpeed'].mean(), 2)
n_dogs    = (df['Type'] == 'Perro').sum()
n_cats    = (df['Type'] == 'Gato').sum()


def cat_vs_adoption(col_name, top=None):
    valid = df[[col_name, 'AdoptionSpeed']].dropna()
    if top:
        tc    = valid[col_name].value_counts().head(top).index
        valid = valid[valid[col_name].isin(tc)]
    ma = valid.groupby(col_name)['AdoptionSpeed'].mean().sort_values()
    return ma.index.tolist(), ma.values


def num_vs_adoption(col_name):
    valid = df[[col_name, 'AdoptionSpeed']].dropna()
    mb    = valid.groupby('AdoptionSpeed')[col_name].mean()
    lbs   = [ADOPTION_LABELS.get(i, str(i)) for i in mb.index]
    return lbs, mb.values


# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR = html.Div([
    html.Div([
        html.Div('🐾', style={'fontSize': '2rem', 'lineHeight': '1'}),
        html.H5('PetFinder EDA', style={'color': 'white', 'fontWeight': '700',
                                         'margin': '0.4rem 0 0', 'fontSize': '1rem'}),
        html.P('Adoption Prediction', style={'color': '#8892b0', 'fontSize': '0.72rem',
                                              'margin': '0', 'letterSpacing': '0.5px'}),
    ], style={'padding': '1.75rem 1.5rem 1.25rem',
               'borderBottom': '1px solid rgba(255,255,255,0.08)'}),

    html.Div([
        html.P('DATASET', style={'color': '#8892b0', 'fontSize': '0.65rem', 'fontWeight': '700',
                                  'letterSpacing': '1.5px', 'margin': '0 0 0.75rem'}),
        *[html.Div([
            html.Span(label, style={'color': '#94a3b8', 'fontSize': '0.78rem'}),
            html.Span(str(val), style={'color': 'white', 'fontWeight': '600',
                                        'fontSize': '0.85rem', 'float': 'right'}),
        ], style={'marginBottom': '0.5rem', 'overflow': 'hidden'})
        for label, val in [
            ('Registros',      f'{len(df):,}'),
            ('Variables',      str(df.shape[1] - 1)),
            ('Perros',         f'{n_dogs:,}'),
            ('Gatos',          f'{n_cats:,}'),
            ('Con metadata',   f"{df['has_face'].notna().sum():,}" if HAS_META else 'N/A'),
            ('Con sentimiento', f"{df['doc_score'].notna().sum():,}" if HAS_SENT else 'N/A'),
        ]],
    ], style={'padding': '1.25rem 1.5rem',
               'borderBottom': '1px solid rgba(255,255,255,0.08)'}),

    html.Div([
        html.P('SECCIONES', style={'color': '#8892b0', 'fontSize': '0.65rem', 'fontWeight': '700',
                                    'letterSpacing': '1.5px', 'margin': '0 0 0.75rem'}),
        *[html.Div(
            html.Span(icon + '  ' + label, style={'color': '#94a3b8', 'fontSize': '0.82rem'}),
            style={'marginBottom': '0.6rem', 'paddingLeft': '0.25rem'}
        )
        for icon, label in [
            ('📊', 'Distribución'),
            ('🔗', 'Asociación'),
            ('✔', 'Significación'),
            ('📝', 'Texto & Sentiment'),
            ('🤖', 'Modelo'),
        ]],
    ], style={'padding': '1.25rem 1.5rem',
               'borderBottom': '1px solid rgba(255,255,255,0.08)'}),

    html.Div([
        html.P('AdoptionSpeed', style={'color': '#8892b0', 'fontSize': '0.65rem',
                                        'fontWeight': '700', 'letterSpacing': '1.5px',
                                        'margin': '0 0 0.5rem'}),
        *[html.Div([
            html.Div(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                            'borderRadius': '50%', 'background': PALETTE[i],
                            'marginRight': '6px', 'verticalAlign': 'middle'}),
            html.Span(lbl, style={'color': '#94a3b8', 'fontSize': '0.75rem'}),
        ]) for i, lbl in enumerate(ADOPTION_ORDER)],
    ], style={'padding': '1.25rem 1.5rem'}),

    html.Div([
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)', 'margin': '0 0 0.75rem'}),
        html.P('Kaggle Competition', style={'color': '#4b5563', 'fontSize': '0.68rem', 'margin': '0'}),
        html.P('petfinder-adoption-prediction',
               style={'color': '#374151', 'fontSize': '0.65rem', 'margin': '0', 'wordBreak': 'break-all'}),
    ], style={'padding': '0 1.5rem 1.5rem', 'marginTop': 'auto'}),
], style={
    'background': SIDEBAR_BG,
    'minHeight': '100vh',
    'width': SIDEBAR_W,
    'position': 'fixed',
    'left': '0', 'top': '0',
    'display': 'flex', 'flexDirection': 'column',
    'zIndex': '1000',
    'overflowY': 'auto',
})

KPI_ROW = dbc.Row([
    dbc.Col(kpi_card('Total Registros',   f'{len(df):,}',    'train.csv',                     '🐾', C_BLUE),   md=3),
    dbc.Col(kpi_card('Adoptados ≤ 30d',   f'{pct_30d}%',     'AdoptionSpeed 0-2',              '✔', C_GREEN),  md=3),
    dbc.Col(kpi_card('Speed Promedio',    f'{avg_speed}',    '0 = mismo día, 4 = no adoptado', '⚡', C_ORANGE), md=3),
    dbc.Col(kpi_card('Registros totales', f'{n_dogs+n_cats:,}', f'🐕 {n_dogs:,}  🐈 {n_cats:,}', '📸', C_PURPLE), md=3),
], className='g-3 mb-4')

INFO_PANEL = info_panel('Resumen del Dataset', [
    ('Tipo modal',       df['Type'].mode().iloc[0] if not df['Type'].mode().empty else 'N/A'),
    ('Edad mediana',     f"{int(df['Age'].median())} meses"),
    ('Fee mediana',      f"RM {int(df['Fee'].median())}"),
    ('Fotos promedio',   f"{df['PhotoAmt'].mean():.1f}"),
    ('Estado más común', df['State'].mode().iloc[0] if not df['State'].mode().empty else 'N/A'),
    ('Speed más común',  ADOPTION_LABELS.get(int(df['AdoptionSpeed'].mode().iloc[0]), 'N/A')),
])


# ── Tab 1: Distribución ───────────────────────────────────────────────────────
def build_tab1():
    df_disp  = df.drop(columns=['PetID'], errors='ignore')
    children = []

    children.append(dbc.Row([
        dbc.Col([
            card([
                sub_title('Tipos de datos por variable'),
                dtable(pd.DataFrame({
                    'Variable':   df_disp.columns,
                    'Tipo':       df_disp.dtypes.astype(str).values,
                    'No nulos':   df_disp.notna().sum().values,
                    '% completo': (df_disp.notna().mean() * 100).round(1).values,
                }))
            ])
        ], md=7),
        dbc.Col([
            card([
                sub_title('Valores nulos'),
                dtable(
                    pd.DataFrame({
                        'Variable': df_disp.columns,
                        'Nulos':    df_disp.isnull().sum().values,
                        '% nulo':   (df_disp.isnull().mean() * 100).round(2).values,
                    }).query('Nulos > 0').sort_values('Nulos', ascending=False).reset_index(drop=True)
                    if df_disp.isnull().any().any() else
                    pd.DataFrame({'Mensaje': ['Sin valores nulos']})
                )
            ]),
            card([sub_title('Missingno — Completitud'), html.Div(id='msno-placeholder')]),
        ], md=5),
    ]))

    plt.figure(figsize=(10, 5))
    msno.matrix(df_disp, sparkline=False, color=(0.12, 0.47, 0.71))
    msno_mat = mpl_to_b64(plt.gcf(), 'msno_matrix')
    plt.close('all')
    plt.figure(figsize=(10, 4))
    msno.bar(df_disp, color='#3b82f6', fontsize=8)
    msno_bar_img = mpl_to_b64(plt.gcf(), 'msno_bar')
    plt.close('all')

    children[-1].children[1].children[1].children[-1] = html.Img(src=msno_mat, style={'width': '100%'})
    children.append(card([sub_title('Missingno — Barras de completitud'),
                           html.Img(src=msno_bar_img, style={'width': '100%'})]))

    num_cols_s = df_disp.select_dtypes(include=[np.number]).columns.tolist()
    stat_rows  = []
    for cn in num_cols_s:
        s = df_disp[cn].dropna()
        if len(s) == 0:
            continue
        moda = s.mode()
        stat_rows.append({'Variable': cn, 'Media': round(float(s.mean()), 4),
                           'Mediana': round(float(s.median()), 4),
                           'Mínimo': round(float(s.min()), 4), 'Máximo': round(float(s.max()), 4),
                           'Moda': round(float(moda.iloc[0]), 4) if not moda.empty else 'N/A',
                           'Desv. Est.': round(float(s.std()), 4)})
    children.append(card([sub_title('Estadísticas descriptivas — Variables numéricas'),
                           dtable(pd.DataFrame(stat_rows))]))

    children.append(section_title('Variable Objetivo — Velocidad de Adopción'))
    adop_counts = df['AdoptionSpeed'].map(ADOPTION_LABELS).value_counts()
    adop_ord    = adop_counts.reindex(ADOPTION_ORDER, fill_value=0)
    children.append(card([
        graph(px_area_bar(adop_ord.index.tolist(), adop_ord.values.tolist(),
                           'Distribución de AdoptionSpeed', 'Velocidad de adopción',
                           'N° de mascotas', color=C_BLUE, height=350), 'dist_adoption_speed')
    ]))

    children.append(section_title('Variables generales'))
    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df['Type']).index.tolist(), top_n(df['Type']).values.tolist(),
                                    'Tipo de animal', yt='Cantidad', color=C_BLUE))]), md=6),
        dbc.Col(card([graph(px_bar(top_n(df['Gender']).index.tolist(), top_n(df['Gender']).values.tolist(),
                                    'Género', yt='Cantidad', color=C_GREEN))]), md=6),
    ], className='g-3'))

    children.append(dbc.Row([
        dbc.Col(card([graph(px_box_chart(df['Age'],      'Edad (meses)',         'Meses',    C_BLUE))]),   md=4),
        dbc.Col(card([graph(px_box_chart(df['Fee'],      'Tarifa de adopción',   'Monto',    C_ORANGE))]), md=4),
        dbc.Col(card([graph(px_box_chart(df['Quantity'], 'Cantidad de animales', 'Cantidad', C_PURPLE))]), md=4),
    ], className='g-3'))

    children.append(dbc.Row([
        dbc.Col(card([graph(px_box_chart(df['PhotoAmt'], 'N° de fotos',  'Fotos',  C_GREEN))]), md=6),
        dbc.Col(card([graph(px_box_chart(df['VideoAmt'], 'N° de videos', 'Videos', C_TEAL))]),  md=6),
    ], className='g-3'))

    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df['MaturitySize']).index.tolist(),
                                    top_n(df['MaturitySize']).values.tolist(),
                                    'Tamaño de madurez', yt='Cantidad', color=C_INDIGO))]), md=6),
        dbc.Col(card([graph(px_bar(top_n(df['FurLength']).index.tolist(),
                                    top_n(df['FurLength']).values.tolist(),
                                    'Longitud del pelaje', yt='Cantidad', color=C_AMBER))]), md=6),
    ], className='g-3'))

    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df['Health']).index.tolist(),
                                    top_n(df['Health']).values.tolist(),
                                    'Estado de salud', yt='Cantidad', color=C_RED))]), md=5),
        dbc.Col(card([graph(px_bar(top_n(df['State']).index.tolist(),
                                    top_n(df['State']).values.tolist(),
                                    'Estado (Malasia)', yt='', color=C_BLUE,
                                    horizontal=True, height=400))]), md=7),
    ], className='g-3'))

    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df['Breed1']).index.tolist(),
                                    top_n(df['Breed1']).values.tolist(),
                                    'Raza principal (Top 9)', yt='', color=C_GREEN,
                                    horizontal=True, height=380))]), md=6),
        dbc.Col(card([graph(px_bar(top_n(df['Breed2']).index.tolist(),
                                    top_n(df['Breed2']).values.tolist(),
                                    'Raza secundaria (Top 9)', yt='', color=C_PURPLE,
                                    horizontal=True, height=380))]), md=6),
    ], className='g-3'))

    children.append(divider())
    children.append(section_title('Grupo: Color'))
    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df[cn]).index.tolist(), top_n(df[cn]).values.tolist(),
                                    f'Color — {cn}', yt='Cantidad', color=PALETTE[i]))]), md=4)
        for i, cn in enumerate(['Color_1', 'Color_2', 'Color_3'])
    ], className='g-3'))

    children.append(divider())
    children.append(section_title('Grupo: Salud'))
    children.append(dbc.Row([
        dbc.Col(card([graph(px_bar(top_n(df[cn]).index.tolist(), top_n(df[cn]).values.tolist(),
                                    lbl, yt='Cantidad', color=PALETTE[i + 2]))]), md=4)
        for i, (cn, lbl) in enumerate([
            ('Vaccinated', 'Vacunado'), ('Dewormed', 'Desparasitado'), ('Sterilized', 'Esterilizado')
        ])
    ], className='g-3'))

    if HAS_META:
        children.append(divider())
        children.append(section_title('Grupo: Metadata de imágenes'))
        crop_hi  = (df['crop_confidence'] > 0.8).mean() * 100
        face_pct = df['has_face'].mean() * 100
        text_pct = df['has_text'].mean() * 100
        fig_ann  = go.Figure()
        ann_labels = ['CropHint > 0.8', 'Anotación facial', 'Anotación de texto']
        ann_vals   = [round(crop_hi, 1), round(face_pct, 1), round(text_pct, 1)]
        fig_ann.add_trace(go.Bar(x=ann_vals, y=ann_labels, orientation='h',
                                  marker_color=[C_BLUE, C_GREEN, C_ORANGE],
                                  marker_line_width=0, text=[f'{v:.1f}%' for v in ann_vals],
                                  textposition='outside'))
        fig_ann.update_layout(**chart_layout('Presencia de anotaciones en imágenes',
                                              xt='% de mascotas', height=280, showlegend=False))
        fig_ann.update_yaxes(showgrid=False)
        fig_ann.update_xaxes(gridcolor='#f0f4f8', range=[0, max(ann_vals) * 1.25])

        children.append(dbc.Row([
            dbc.Col(card([graph(px_box_chart(df['n_labels_img'],    'Etiquetas por imagen',  'N° etiquetas', C_BLUE))]),   md=3),
            dbc.Col(card([graph(px_box_chart(df['avg_label_score'], 'Score prom. etiquetas', 'Score',        C_GREEN))]),  md=3),
            dbc.Col(card([graph(px_box_chart(df['max_pixelFraction'], 'Fracción píxeles dom.', 'Fracción',   C_ORANGE))]), md=3),
            dbc.Col(card([graph(px_box_chart(df['crop_confidence'], 'Confianza de recorte',  'Confianza',    C_PURPLE))]), md=3),
        ], className='g-3'))
        children.append(card([graph(fig_ann, 'pct_annotations')]))

        valid_rgb = df[['dom_R', 'dom_G', 'dom_B']].dropna()
        if len(valid_rgb) > 0:
            ar = valid_rgb['dom_R'].mean() / 255
            ag = valid_rgb['dom_G'].mean() / 255
            ab = valid_rgb['dom_B'].mean() / 255
            plt.figure(figsize=(3, 1.5))
            plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, color=(ar, ag, ab)))
            plt.xlim(0, 1); plt.ylim(0, 1); plt.axis('off')
            plt.title(f'Color dominante prom. — RGB({int(ar*255)}, {int(ag*255)}, {int(ab*255)})', fontsize=9)
            plt.tight_layout()
            rgb_img = mpl_to_b64(plt.gcf(), 'avg_dominant_color')
            children.append(dbc.Row([
                dbc.Col(card([sub_title('Color promedio del dataset'),
                               html.Img(src=rgb_img, style={'width': '100%'})]), md=3),
                dbc.Col(md=9),
            ], className='g-3'))

        if not labels_df.empty:
            lbl_freq = labels_df['description'].value_counts().head(20)
            top15    = labels_df['description'].value_counts().head(15).index.tolist()
            lbl_st   = (labels_df[labels_df['description'].isin(top15)]
                        .groupby('description')
                        .agg(score_mean=('score', 'mean'), topicality_mean=('topicality', 'mean'))
                        .reindex(top15))
            children.append(dbc.Row([
                dbc.Col(card([sub_title('Top 20 etiquetas más frecuentes'),
                               graph(px_bar(lbl_freq.index.tolist(), lbl_freq.values.tolist(),
                                             '', yt='N° apariciones', color=C_BLUE,
                                             horizontal=True, height=450), 'top20_labels')]), md=6),
                dbc.Col(card([sub_title('Score vs Topicality — Top 15'),
                               graph(px_grouped_bar(lbl_st.index.tolist(),
                                                     lbl_st['score_mean'].tolist(),
                                                     lbl_st['topicality_mean'].tolist(),
                                                     'Score', 'Topicality', '',
                                                     xt='Etiqueta', yt='Valor promedio',
                                                     height=450), 'score_vs_topicality')]), md=6),
            ], className='g-3'))

            lbl_wc = labels_df.groupby('description')['score'].mean().to_dict()
            if lbl_wc:
                wc = WordCloud(background_color='white', width=900, height=380,
                               colormap='Blues', prefer_horizontal=0.85, min_font_size=8)
                wc.generate_from_frequencies(lbl_wc)
                fig_wc, ax_wc = plt.subplots(figsize=(11, 4.5))
                ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
                plt.tight_layout()
                children.append(card([sub_title('Nube de palabras — Etiquetas (score promedio)'),
                                       html.Img(src=mpl_to_b64(fig_wc, 'wc_labels'), style={'width': '100%'})]))

    if HAS_SENT:
        children.append(divider())
        children.append(section_title('Grupo: Sentimientos'))
        children.append(dbc.Row([
            dbc.Col(card([graph(px_box_chart(df['doc_score'],           'Score global',       'Score (-1 a +1)', C_BLUE))]),   md=3),
            dbc.Col(card([graph(px_box_chart(df['doc_magnitude'],        'Magnitud global',    'Magnitud',        C_GREEN))]),  md=3),
            dbc.Col(card([graph(px_box_chart(df['n_sentences'],          'N° de oraciones',    'Oraciones',       C_ORANGE))]), md=3),
            dbc.Col(card([graph(px_box_chart(df['sentence_score_range'], 'Variación emocional','Rango de score',  C_PURPLE))]), md=3),
        ], className='g-3'))
        children.append(dbc.Row([
            dbc.Col(card([graph(px_bar(top_n(df['sentiment_class']).index.tolist(),
                                        top_n(df['sentiment_class']).values.tolist(),
                                        'Clasificación de sentimiento', yt='Cantidad', color=C_TEAL))]), md=5),
            dbc.Col(card([sub_title('Score de sentimiento por tipo de animal'),
                           graph(px.box(df[['Type', 'doc_score']].dropna(),
                                         x='Type', y='doc_score', color='Type',
                                         color_discrete_sequence=PALETTE,
                                         labels={'Type': 'Tipo de animal', 'doc_score': 'Score'},
                                        ).update_layout(**chart_layout('', height=320, showlegend=False)))]), md=7),
        ], className='g-3'))

        if not ent_df.empty:
            ent_imgs = []
            for et, label_es, col_idx in [('PERSON', 'Personas', 0), ('LOCATION', 'Lugares', 1), ('OTHER', 'Otros', 2)]:
                sub = ent_df[ent_df['type'] == et]
                if len(sub) > 0:
                    top_e = sub.groupby('name')['salience'].mean().sort_values(ascending=False).head(10)
                    ent_imgs.append(dbc.Col(card([graph(px_bar(
                        top_e.index.tolist(), top_e.values.tolist(),
                        f'Entidades: {label_es}', yt='Salience prom.',
                        color=PALETTE[col_idx + 2], horizontal=True, height=350))]), md=4))
            if ent_imgs:
                children.append(dbc.Row(ent_imgs, className='g-3'))

            ent_wc = ent_df.groupby('name')['salience'].mean().to_dict()
            if ent_wc:
                wc2 = WordCloud(background_color='white', width=900, height=360,
                                colormap='Greys', prefer_horizontal=0.8)
                wc2.generate_from_frequencies(ent_wc)
                fig_wc2, ax_wc2 = plt.subplots(figsize=(11, 4.5))
                ax_wc2.imshow(wc2, interpolation='bilinear'); ax_wc2.axis('off')
                plt.tight_layout()
                children.append(card([sub_title('Nube de palabras — Entidades (salience promedio)'),
                                       html.Img(src=mpl_to_b64(fig_wc2, 'wc_entities'), style={'width': '100%'})]))
    return children


# ── Tab 2: Asociación ─────────────────────────────────────────────────────────
def build_tab2():
    children = []
    num_all = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_m  = df[num_all].corr()

    children.append(card([sub_title('Tabla de correlación — Variables numéricas'),
                           dtable(corr_m.round(3).reset_index().rename(columns={'index': 'Variable'}))]))
    children.append(card([sub_title('Heatmap de correlación'),
                           graph(px_heatmap_corr(corr_m), 'heatmap_correlacion')]))
    children.append(divider())
    children.append(section_title('Asociación de variables con AdoptionSpeed'))

    def assoc_bar(x_labels, y_vals, title, xt='', yl='', col=C_BLUE):
        return graph(px_bar(x_labels, y_vals.tolist(), title, xt=xt, yt=yl, color=col, height=320))

    children.append(section_title('Variables generales'))
    num_pairs = [('Age','Edad prom. (meses)', C_BLUE), ('Fee','Tarifa prom.', C_ORANGE),
                 ('Quantity','Cantidad prom.', C_PURPLE), ('PhotoAmt','Fotos prom.', C_GREEN),
                 ('VideoAmt','Videos prom.', C_TEAL)]
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{cn} por AdoptionSpeed',
                                 'Velocidad de adopción', yl, col)]), md=4)
        for cn, yl, col in num_pairs[:3]
    ], className='g-3'))
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{cn} por AdoptionSpeed',
                                 'Velocidad de adopción', yl, col)]), md=6)
        for cn, yl, col in num_pairs[3:]
    ], className='g-3'))

    cat_pairs = [('Type','Tipo de animal', C_BLUE), ('Gender','Género', C_GREEN),
                 ('MaturitySize','Tamaño madurez', C_PURPLE), ('FurLength','Pelaje', C_ORANGE),
                 ('Health','Salud', C_RED)]
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*cat_vs_adoption(cn), f'AdoptionSpeed — {lbl}',
                                 lbl, 'AdoptionSpeed prom.', col)]), md=4)
        for cn, lbl, col in cat_pairs[:3]
    ], className='g-3'))
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*cat_vs_adoption(cn), f'AdoptionSpeed — {lbl}',
                                 lbl, 'AdoptionSpeed prom.', col)]), md=6)
        for cn, lbl, col in cat_pairs[3:]
    ], className='g-3'))

    for cn, lbl, topk, col in [('State','Estado',10,C_TEAL), ('Breed1','Raza principal',9,C_INDIGO),
                                 ('Breed2','Raza secundaria',9,C_AMBER)]:
        xl, yv = cat_vs_adoption(cn, top=topk)
        children.append(card([assoc_bar(xl, yv, f'AdoptionSpeed — {lbl} (Top {topk})',
                                         lbl, 'AdoptionSpeed prom.', col)]))

    children.append(divider())
    children.append(section_title('Grupo: Color'))
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*cat_vs_adoption(cn), f'AdoptionSpeed — Color {cn}',
                                 cn, 'AdoptionSpeed prom.', PALETTE[i])]), md=4)
        for i, cn in enumerate(['Color_1', 'Color_2', 'Color_3'])
    ], className='g-3'))

    children.append(divider())
    children.append(section_title('Grupo: Salud'))
    children.append(dbc.Row([
        dbc.Col(card([assoc_bar(*cat_vs_adoption(cn), f'AdoptionSpeed — {lbl}',
                                 lbl, 'AdoptionSpeed prom.', PALETTE[i + 3])]), md=4)
        for i, (cn, lbl) in enumerate([
            ('Vaccinated','Vacunado'), ('Dewormed','Desparasitado'), ('Sterilized','Esterilizado')
        ])
    ], className='g-3'))

    if HAS_META:
        children.append(divider())
        children.append(section_title('Grupo: Metadata de imágenes'))
        meta_num = [('n_labels_img','Etiquetas por imagen',C_BLUE),
                    ('avg_label_score','Score prom. etiquetas',C_GREEN),
                    ('max_pixelFraction','Fracción píxeles dom.',C_ORANGE),
                    ('crop_confidence','Confianza de recorte',C_PURPLE)]
        children.append(dbc.Row([
            dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{lbl} por AdoptionSpeed',
                                     'Velocidad de adopción', lbl, col)]), md=6)
            for cn, lbl, col in meta_num[:2]
        ], className='g-3'))
        children.append(dbc.Row([
            dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{lbl} por AdoptionSpeed',
                                     'Velocidad de adopción', lbl, col)]), md=6)
            for cn, lbl, col in meta_num[2:]
        ], className='g-3'))

    if HAS_SENT:
        children.append(divider())
        children.append(section_title('Grupo: Sentimientos'))
        sent_pairs = [('doc_score','Score global',C_BLUE), ('doc_magnitude','Magnitud',C_GREEN),
                      ('n_sentences','N° de oraciones',C_ORANGE),
                      ('sentence_score_range','Variación emoc.',C_PURPLE)]
        children.append(dbc.Row([
            dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{lbl} por AdoptionSpeed',
                                     'Velocidad de adopción', lbl, col)]), md=6)
            for cn, lbl, col in sent_pairs[:2]
        ], className='g-3'))
        children.append(dbc.Row([
            dbc.Col(card([assoc_bar(*num_vs_adoption(cn), f'{lbl} por AdoptionSpeed',
                                     'Velocidad de adopción', lbl, col)]), md=6)
            for cn, lbl, col in sent_pairs[2:]
        ], className='g-3'))

        if 'sentiment_class' in df.columns:
            xl, yv = cat_vs_adoption('sentiment_class')
            children.append(card([assoc_bar(xl, yv, 'AdoptionSpeed — Clase de sentimiento',
                                             'Clasificación', 'AdoptionSpeed prom.', C_TEAL)]))
            valid_sc = df[['sentiment_class', 'AdoptionSpeed']].dropna()
            cross    = pd.crosstab(valid_sc['sentiment_class'],
                                   valid_sc['AdoptionSpeed'].map(ADOPTION_LABELS),
                                   normalize='index') * 100
            cross    = cross.reindex(columns=ADOPTION_ORDER, fill_value=0)
            children.append(card([sub_title('Cruce: Clasificación de sentimiento × AdoptionSpeed'),
                                   graph(px_multibar_stacked(
                                       cross, 'Distribución de AdoptionSpeed por sentimiento',
                                       xt='Velocidad de adopción', yt='% dentro de la clase'),
                                       'cross_sentiment_adoption')]))
    return children


# ── Tab 3: Significación ──────────────────────────────────────────────────────
def build_tab3():
    children = []
    print('  Calculando asociaciones…')
    pairs_df  = compute_associations(df)
    THRESHOLD = 0.30
    sig_pairs = pairs_df[pairs_df['medida'] > THRESHOLD].copy()
    work_pairs = sig_pairs if not sig_pairs.empty else pairs_df.head(10)

    children.append(dbc.Alert(
        f'{"✔  " + str(len(sig_pairs)) + " pares con |coeficiente| > 0.30 encontrados." if not sig_pairs.empty else "⚠  Sin pares con |coeficiente| > 0.30. Se muestran los 10 más fuertes."}',
        color='success' if not sig_pairs.empty else 'warning',
        style={'borderRadius': '12px', 'fontWeight': '600', 'marginBottom': '1rem'},
    ))

    results = []
    for _, row_p in work_pairs.iterrows():
        v1, v2, tipo = row_p['var1'], row_p['var2'], row_p['tipo']
        medida       = row_p['medida_raw']
        test_name    = stat_v = p_v = np.nan

        if tipo == 'num-num':
            test_name = 'Pearson'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10:
                try:
                    stat_v, p_v = pearsonr(valid[v1], valid[v2])
                except Exception:
                    pass
        elif tipo == 'cat-num':
            test_name = 'Kruskal-Wallis'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10 and valid[v1].nunique() >= 2:
                groups = [valid[v2][valid[v1] == c].values for c in valid[v1].unique()
                          if len(valid[v2][valid[v1] == c]) > 0]
                if len(groups) >= 2:
                    try:
                        stat_v, p_v = kruskal(*groups)
                    except Exception:
                        pass
        elif tipo == 'cat-cat':
            test_name = 'Chi-cuadrado'
            valid = df[[v1, v2]].dropna()
            if len(valid) >= 10:
                try:
                    ct = pd.crosstab(valid[v1], valid[v2])
                    stat_v, p_v, _, _ = chi2_contingency(ct)
                except Exception:
                    pass

        def fmt(x, dec=4):
            try:
                v = float(x)
                return str(round(v, dec)) if not np.isnan(v) else 'N/A'
            except (TypeError, ValueError):
                return 'N/A'

        def fmt_p(x):
            try:
                v = float(x)
                return f'{v:.4e}' if not np.isnan(v) else 'N/A'
            except (TypeError, ValueError):
                return 'N/A'

        try:
            interp = 'Significativo ✔' if float(p_v) < 0.05 else 'No significativo ✘'
        except (TypeError, ValueError):
            interp = 'N/A'

        tipo_leg = {'num-num': 'Num → Num', 'cat-num': 'Cat → Num', 'cat-cat': 'Cat → Cat'}.get(tipo, tipo)
        results.append({'Variable 1': v1, 'Variable 2': v2, 'Tipo de par': tipo_leg,
                         'Test': test_name, 'Coeficiente': fmt(medida),
                         'Estadístico': fmt(stat_v), 'p-valor': fmt_p(p_v),
                         'Interpretación': interp})

    res_df = pd.DataFrame(results)
    n_sig  = (res_df['Interpretación'].str.startswith('Significativo')).sum()
    children.append(card([
        sub_title('Tabla de resultados'),
        dtable(res_df, page_size=20),
        html.P(f'{n_sig} de {len(res_df)} pares son estadísticamente significativos (α = 0.05)',
               style={'fontWeight': '600', 'color': C_GREEN, 'marginTop': '0.75rem', 'marginBottom': '0'}),
    ]))

    if not sig_pairs.empty:
        tipo_counts = sig_pairs['tipo'].map({'num-num': 'Num → Num',
                                              'cat-num': 'Cat → Num',
                                              'cat-cat': 'Cat → Cat'}).value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=tipo_counts.index.tolist(), values=tipo_counts.values.tolist(),
            hole=0.5, marker_colors=PALETTE[:3],
            textinfo='label+percent', insidetextorientation='radial',
        ))
        fig_pie.update_layout(**chart_layout('Tipos de pares significativos', height=320))

        fig_hist = go.Figure(go.Histogram(
            x=sig_pairs['medida'].values, nbinsx=20,
            marker_color=C_BLUE, marker_line_width=0, opacity=0.88,
        ))
        fig_hist.add_vline(x=THRESHOLD, line_color=C_RED, line_width=2, line_dash='dash')
        fig_hist.update_layout(**chart_layout('Distribución de coeficientes (|coef| > 0.30)',
                                               xt='|Coeficiente|', yt='N° de pares',
                                               height=320, showlegend=False))
        children.append(dbc.Row([
            dbc.Col(card([graph(fig_hist, 'hist_asociaciones')]), md=7),
            dbc.Col(card([graph(fig_pie,  'pie_tipos_pares')]),   md=5),
        ], className='g-3'))
    return children


# ── Build tabs ────────────────────────────────────────────────────────────────
print('  Tab 1 — Distribución…')
tab1_content = build_tab1()
print('  Tab 2 — Asociación…')
tab2_content = build_tab2()
print('  Tab 3 — Significación…')
tab3_content = build_tab3()
print('  Tab 4 — Texto & Sentiment…')

# ── Tab 4: Texto & Sentiment ──────────────────────────────────────────────────
def build_tab_texto():
    try:
        sent_df = pd.read_csv(BASE / 'train_sentiment_features.csv')
        meta_df = pd.read_csv(BASE / 'train_metadata_features.csv')
        df_s = DF.merge(sent_df[['PetID','sentiment_score','sentiment_magnitude','n_sentences','language']], on='PetID', how='left')
        df_s = df_s.merge(meta_df[['PetID','avg_label_score','n_labels','crop_confidence']], on='PetID', how='left')
        df_s['desc_length'] = df_s['Description'].fillna('').apply(len)
        df_s['AdoptionLabel'] = df_s['AdoptionSpeed'].map(ADOPTION_LABELS)
        df_s = df_s.fillna(0)
    except Exception as e:
        return html.P(f'Error cargando datos de texto: {e}', style={'color': C_RED})

    fig_sent = px.bar(
        df_s.groupby('AdoptionLabel')['sentiment_score'].mean().reset_index(),
        x='AdoptionLabel', y='sentiment_score', color='AdoptionLabel',
        color_discrete_sequence=PALETTE, template='plotly_white',
        title='Sentiment Score promedio por clase de adopción',
    )
    fig_sent.update_layout(**chart_layout(xt='Clase', yt='Score (−1 a +1)', showlegend=False))

    fig_box = px.box(df_s, x='AdoptionLabel', y='sentiment_score',
                     color='AdoptionLabel', color_discrete_sequence=PALETTE,
                     template='plotly_white',
                     title='Distribución de Sentiment Score por clase')
    fig_box.update_layout(**chart_layout(xt='Clase', yt='Score', showlegend=False))

    fig_desc = px.bar(
        df_s.groupby('AdoptionLabel')['desc_length'].mean().reset_index(),
        x='AdoptionLabel', y='desc_length', color='AdoptionLabel',
        color_discrete_sequence=PALETTE, template='plotly_white',
        title='Longitud promedio de descripción por clase',
    )
    fig_desc.update_layout(**chart_layout(xt='Clase', yt='Caracteres', showlegend=False))

    lang_counts = df_s['language'].value_counts().reset_index()
    lang_counts.columns = ['Idioma', 'Cantidad']
    fig_lang = px.pie(lang_counts.head(6), values='Cantidad', names='Idioma',
                      title='Distribución de idiomas en las descripciones',
                      color_discrete_sequence=PALETTE, template='plotly_white')
    fig_lang.update_traces(textposition='inside', textinfo='percent+label')
    fig_lang.update_layout(**chart_layout(showlegend=True))

    fig_img = px.bar(
        df_s.groupby('AdoptionLabel')['avg_label_score'].mean().reset_index(),
        x='AdoptionLabel', y='avg_label_score', color='AdoptionLabel',
        color_discrete_sequence=PALETTE, template='plotly_white',
        title='Calidad de imagen promedio (Google Vision) por clase',
    )
    fig_img.update_layout(**chart_layout(xt='Clase', yt='Score (0–1)', showlegend=False))

    fig_mag = px.bar(
        df_s.groupby('AdoptionLabel')['sentiment_magnitude'].mean().reset_index(),
        x='AdoptionLabel', y='sentiment_magnitude', color='AdoptionLabel',
        color_discrete_sequence=PALETTE, template='plotly_white',
        title='Sentiment Magnitude promedio por clase (intensidad emocional)',
    )
    fig_mag.update_layout(**chart_layout(xt='Clase', yt='Magnitud', showlegend=False))

    return html.Div([
        section_title('Análisis de Texto, Sentimiento e Imágenes'),
        html.P('Fuentes: Google Natural Language API · Google Vision API · 14.442 JSONs de sentiment · 58.311 JSONs de metadata',
               style={'color': TEXT_MUTED, 'fontSize': '0.82rem', 'marginBottom': '1.2rem'}),
        dbc.Row([
            dbc.Col([kpi_card('Sentiment promedio', f'{df_s["sentiment_score"].mean():.3f}',
                              'Escala −1 (neg) a +1 (pos)', '💬', C_GREEN)], md=3),
            dbc.Col([kpi_card('Desc. promedio', f'{int(df_s["desc_length"].mean())} chars',
                              f'Sin desc: {(df_s["desc_length"]==0).sum()} mascotas', '📝', C_BLUE)], md=3),
            dbc.Col([kpi_card('Calidad imagen', f'{df_s["avg_label_score"].mean():.3f}',
                              'Score Google Vision (0–1)', '📷', C_ORANGE)], md=3),
            dbc.Col([kpi_card('Magnitude promedio', f'{df_s["sentiment_magnitude"].mean():.3f}',
                              'Intensidad emocional del texto', '❤️', C_PURPLE)], md=3),
        ], className='g-3', style={'marginBottom': '1.5rem'}),
        dbc.Row([
            dbc.Col([card([graph(fig_sent)])], md=6),
            dbc.Col([card([graph(fig_box)])],  md=6),
        ], className='g-3'),
        dbc.Row([
            dbc.Col([card([graph(fig_desc)])], md=6),
            dbc.Col([card([graph(fig_lang)])], md=6),
        ], className='g-3'),
        dbc.Row([
            dbc.Col([card([graph(fig_img)])],  md=6),
            dbc.Col([card([graph(fig_mag)])],  md=6),
        ], className='g-3'),
        divider(),
        section_title('Hallazgos clave del análisis multimodal'),
        dbc.Row([
            dbc.Col([card([
                html.P('📌 desc_length es la variable #1 en importancia', style={'fontWeight': '700', 'color': TEXT_PRIMARY, 'marginBottom': '0.3rem'}),
                html.P('Las mascotas con descripciones más largas y emotivas se adoptan más rápido. '
                       'La calidad de la presentación importa tanto como las características físicas del animal.',
                       style={'color': TEXT_MUTED, 'fontSize': '0.85rem'}),
            ])], md=6),
            dbc.Col([card([
                html.P('📌 avg_label_score (calidad de foto) es la variable #2', style={'fontWeight': '700', 'color': TEXT_PRIMARY, 'marginBottom': '0.3rem'}),
                html.P('La calidad de las imágenes según Google Vision supera en importancia a la edad y la raza. '
                       'Fotos claras y bien encuadradas aumentan significativamente las probabilidades de adopción.',
                       style={'color': TEXT_MUTED, 'fontSize': '0.85rem'}),
            ])], md=6),
        ], className='g-3'),
        dbc.Row([
            dbc.Col([card([
                html.P('📌 Mascotas sin foto se adoptan hasta 2x más lento', style={'fontWeight': '700', 'color': TEXT_PRIMARY, 'marginBottom': '0.3rem'}),
                html.P('PhotoAmt = 0 correlaciona fuertemente con AdoptionSpeed = 4 (>100 días). '
                       'Subir al menos una foto es el cambio más impactante que puede hacer un rescatista.',
                       style={'color': TEXT_MUTED, 'fontSize': '0.85rem'}),
            ])], md=6),
            dbc.Col([card([
                html.P('📌 sentiment_magnitude supera al sentiment_score', style={'fontWeight': '700', 'color': TEXT_PRIMARY, 'marginBottom': '0.3rem'}),
                html.P('La intensidad emocional del texto (cuánta emoción transmite) predice mejor que la polaridad '
                       '(positivo/negativo). Descripciones apasionadas funcionan mejor que descripciones neutras.',
                       style={'color': TEXT_MUTED, 'fontSize': '0.85rem'}),
            ])], md=6),
        ], className='g-3'),
        divider(),
        section_title('Prevención de Data Leakage'),
        card([
            html.P('El data leakage ocurre cuando información del futuro o del test contamina el entrenamiento. '
                   'En este proyecto tomamos las siguientes precauciones:', style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '0.8rem'}),
            dbc.Row([
                dbc.Col([
                    html.Ul([
                        html.Li([html.Strong('Target Encoding: '), 'calculado SOLO sobre el fold de train en cada split del CV. '
                                 'El fold de validación y el test nunca participan en el cálculo del encoding.'],
                                style={'color': TEXT_MUTED, 'fontSize': '0.84rem', 'marginBottom': '0.5rem'}),
                        html.Li([html.Strong('Optuna CV: '), 'el optimizador evalúa sobre validación cruzada interna del training set. '
                                 'El test set solo se usa para el reporte final, nunca para optimización.'],
                                style={'color': TEXT_MUTED, 'fontSize': '0.84rem', 'marginBottom': '0.5rem'}),
                    ], style={'paddingLeft': '1.2rem'})
                ], md=6),
                dbc.Col([
                    html.Ul([
                        html.Li([html.Strong('Features de texto/imagen: '), 'parseadas de JSONs pre-computados por Google. '
                                 'No hay target leakage porque son características del perfil publicado, no del resultado de adopción.'],
                                style={'color': TEXT_MUTED, 'fontSize': '0.84rem', 'marginBottom': '0.5rem'}),
                        html.Li([html.Strong('Split estratificado: '), 'train/test mantiene la misma proporción de clases. '
                                 'Seed fijo (42) para reproducibilidad.'],
                                style={'color': TEXT_MUTED, 'fontSize': '0.84rem', 'marginBottom': '0.5rem'}),
                    ], style={'paddingLeft': '1.2rem'})
                ], md=6),
            ]),
        ]),
    ])

tab_texto_content = build_tab_texto()
print('  Tab 5 — Modelo…')

# ── Tab 4: Modelo ─────────────────────────────────────────────────────────────
# Resultados del notebook 04_Tabulares V2 - Ejecutado Roxy
# Actualizar con valores finales cuando termine Optuna + FE v2

# ── Resultados finales v3 (Abril 2026) ──────────────────────────────────────
# Original (profe): 04_Tabulares.ipynb  — Optuna CV sobre 19 features → CV 0.3639
# Ajustado Roxy:    04_Tabulares_V2_SeccionF.ipynb — FE v2 + Optuna → Test 0.3371

_modelos_df = pd.DataFrame({
    'Modelo': [
        'Baseline',
        'FE v1',
        'FE v2 + Optuna',
        'FE v3 + Optuna simple',
        'FE v3 + Optuna CV',
    ],
    'Kappa Test': [0.3133, 0.3231, 0.3371, 0.3595, 0.3381],
    'Kappa Train':[0.5877, 0.4677, 0.4677, 0.6363, 0.4281],
    'Tipo':       ['Original', 'Ajustado Roxy', 'Ajustado Roxy', 'Ajustado Roxy', 'Ajustado Roxy'],
    'Detalle':    ['19 feat · Original', '26 feat · Roxy', '32 feat · Roxy', '39 feat · Roxy', '41 feat · Roxy (mejor gap)'],
})

_fig_kappa = px.bar(
    _modelos_df, x='Modelo', y='Kappa Test',
    title='Comparativa de Modelos — Cohen Kappa (Test)',
    color='Tipo', template='plotly_white',
    color_discrete_map={'Original': C_BLUE, 'Ajustado Roxy': C_GREEN},
    text='Kappa Test',
    hover_data={'Detalle': True, 'Tipo': False},
)
_fig_kappa.update_traces(texttemplate='%{text:.4f}', textposition='outside')
_fig_kappa.update_layout(showlegend=True,
                          yaxis=dict(range=[0, 0.55]),
                          yaxis_title='Cohen Kappa (quadratic)',
                          xaxis=dict(tickangle=-30, tickfont=dict(size=12)),
                          margin=dict(b=80),
                          height=380)

_fig_gap = px.bar(
    _modelos_df, x='Modelo',
    y=['Kappa Train', 'Kappa Test'],
    barmode='group',
    title='Kappa Train vs Test — Análisis de Overfitting',
    template='plotly_white',
    color_discrete_map={'Kappa Train': C_BLUE, 'Kappa Test': C_ORANGE},
)
_fig_gap.update_layout(yaxis=dict(range=[0, 0.8]),
                        yaxis_title='Cohen Kappa (quadratic)',
                        xaxis=dict(tickangle=-30, tickfont=dict(size=12)),
                        margin=dict(b=80),
                        height=380)

_feat_imp_df = pd.DataFrame({
    'Feature':    ['desc_length', 'avg_label_score', 'sentiment_magnitude',
                   'Age_x_PhotoAmt', 'Age', 'n_labels',
                   'PhotoPerAnimal', 'Breed1', 'sentiment_score', 'n_sentences',
                   'Color2', 'PhotoAmt', 'State', 'Color1', 'Breed2'],
    'Importancia':[5500, 5200, 3100, 2800, 2600, 2400,
                   2200, 2000, 1900, 1800,
                   1600, 1500, 1400, 1300, 1200],
    'Tipo': ['Texto/Imagen','Texto/Imagen','Texto/Imagen',
             'FE v2','Original','Texto/Imagen',
             'FE v1','Original','Texto/Imagen','Texto/Imagen',
             'Original','Original','Original','Original','Original'],
}).sort_values('Importancia')

_fig_fi = px.bar(
    _feat_imp_df, x='Importancia', y='Feature', orientation='h',
    title='Feature Importance (Gain) — Top 15 variables (FE v3)',
    color='Tipo', template='plotly_white',
    color_discrete_map={'Original': C_BLUE, 'FE v1': C_GREEN,
                        'FE v2': C_ORANGE, 'Texto/Imagen': C_PURPLE},
)
_fig_fi.update_layout(showlegend=True)

tab_modelo_content = html.Div([
    html.H5('Resultados del Modelado — LightGBM  |  v5 Final (Abril 2026)',
            style={'color': TEXT_PRIMARY, 'fontWeight': '600', 'marginBottom': '0.3rem'}),
    html.P('Autores: Roxana Alberti · Sandra Sschicchi · Fernando Paganini · Baltazar Villanueva · Paula Calviello · Rosana Martinez',
           style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'marginBottom': '1rem'}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P('Kappa Baseline (Original)', style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'margin': 0}),
                    html.H3('0.3133', style={'color': C_BLUE, 'fontWeight': '700', 'margin': 0}),
                    html.P('19 features, sin FE', style={'color': TEXT_MUTED, 'fontSize': '0.75rem', 'margin': 0}),
                ])
            ], style={'borderTop': f'3px solid {C_BLUE}', 'borderRadius': '12px'}),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P('Mejor Kappa (Optimizado Roxy)', style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'margin': 0}),
                    html.H3('0.3595', style={'color': C_GREEN, 'fontWeight': '700', 'margin': 0}),
                    html.P('+0.0462 vs baseline  •  FE v3 + Optuna 50 trials', style={'color': C_GREEN, 'fontSize': '0.75rem', 'margin': 0}),
                ])
            ], style={'borderTop': f'3px solid {C_GREEN}', 'borderRadius': '12px'}),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P('Features — Optimizado Roxy', style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'margin': 0}),
                    html.H3('39', style={'color': C_ORANGE, 'fontWeight': '700', 'margin': 0}),
                    html.P('19 orig + 13 FE + 7 texto/imágenes', style={'color': TEXT_MUTED, 'fontSize': '0.75rem', 'margin': 0}),
                ])
            ], style={'borderTop': f'3px solid {C_ORANGE}', 'borderRadius': '12px'}),
        ], md=4),
    ], className='g-3', style={'marginBottom': '1.5rem'}),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=_fig_kappa)], md=6),
        dbc.Col([dcc.Graph(figure=_fig_gap)],   md=6),
    ], className='g-3'),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=_fig_fi)], md=12),
    ], className='g-3'),
])

print('Iniciando servidor en http://localhost:8050')

TAB_STYLE = {
    'padding': '0.65rem 1.4rem', 'fontWeight': '500', 'color': TEXT_MUTED,
    'borderRadius': '8px 8px 0 0', 'fontFamily': 'Inter, Roboto, Arial',
}
TAB_SELECTED = {
    **TAB_STYLE, 'fontWeight': '700', 'color': TEXT_PRIMARY,
    'borderTop': f'3px solid {C_BLUE}', 'background': CONTENT_BG,
}
CONTENT_PAD = {'padding': '1.5rem 2rem', 'background': CONTENT_BG}

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[
    dbc.themes.FLATLY,
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
])
app.title = 'PetFinder — EDA Dashboard'
server = app.server  # necesario para gunicorn en Render

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, Roboto, Arial, sans-serif; background: ''' + CONTENT_BG + '''; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f1f5f9; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
'''

app.layout = html.Div([
    SIDEBAR,
    html.Div([
        html.Div([
            html.H4('PetFinder — Adoption Prediction EDA',
                    style={'color': TEXT_PRIMARY, 'fontWeight': '700',
                           'margin': '0', 'fontSize': '1.3rem'}),
            html.P('Exploración interactiva de datos para la predicción de velocidad de adopción',
                   style={'color': TEXT_MUTED, 'margin': '0.2rem 0 0', 'fontSize': '0.85rem'}),
        ], style={'marginBottom': '1.5rem'}),
        KPI_ROW,
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(html.Div(tab1_content, style=CONTENT_PAD),
                            label='📊  Distribución', tab_style=TAB_STYLE, active_tab_style=TAB_SELECTED),
                    dbc.Tab(html.Div(tab2_content, style=CONTENT_PAD),
                            label='🔗  Asociación',   tab_style=TAB_STYLE, active_tab_style=TAB_SELECTED),
                    dbc.Tab(html.Div(tab3_content, style=CONTENT_PAD),
                            label='✔  Significación', tab_style=TAB_STYLE, active_tab_style=TAB_SELECTED),
                    dbc.Tab(html.Div(tab_texto_content, style=CONTENT_PAD),
                            label='📝  Texto & Sentiment', tab_style=TAB_STYLE, active_tab_style=TAB_SELECTED),
                    dbc.Tab(html.Div(tab_modelo_content, style=CONTENT_PAD),
                            label='🤖  Modelo',        tab_style=TAB_STYLE, active_tab_style=TAB_SELECTED),
                ], style={'background': 'white', 'borderRadius': '16px 16px 0 0',
                           'boxShadow': '0 1px 4px rgba(0,0,0,0.07)'}),
            ], md=9),
            dbc.Col([
                html.Div(INFO_PANEL, style={'position': 'sticky', 'top': '1.5rem'}),
            ], md=3),
        ], className='g-3'),
    ], style={'marginLeft': SIDEBAR_W, 'padding': '2rem',
               'minHeight': '100vh', 'background': CONTENT_BG}),
], style={'background': CONTENT_BG})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
