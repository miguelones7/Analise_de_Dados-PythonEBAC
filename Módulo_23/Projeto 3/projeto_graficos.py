import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------- #
#     CARREGAR DADOS     #
# ---------------------- #
df = pd.read_csv("ecommerce_estatistica.csv")

# Ajuste de temporada
df['Temporada'] = df['Temporada'].apply(
    lambda x: x if x in ['primavera/verão', 'outono/inverno'] else 'anual'
)

# ---------------------- #
#     FUNÇÕES DE GRÁFICO #
# ---------------------- #

def grafico_histograma(df):
    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import gaussian_kde

    dados = df['Desconto_MinMax'].dropna()

    # --- KDE (mesmo efeito do seaborn) ---
    kde = gaussian_kde(dados)
    x_kde = np.linspace(0, 1, 200)
    y_kde = kde(x_kde)

    # --- HISTOGRAMA ---
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=dados,
        nbinsx=20,
        marker_color='royalblue',
        opacity=0.7,
        name='Histograma'
    ))

    # --- KDE como linha ---
    fig.add_trace(go.Scatter(
        x=x_kde,
        y=y_kde * len(dados) * (1/20),  # normaliza a curva para “encaixar”
        mode='lines',
        line=dict(color='royalblue', width=2),
        name='Densidade'
    ))

    # --- LAYOUT ---
    fig.update_layout(
        title="Gráfico de Histograma<br>Distribuição dos Descontos (%)",
        xaxis=dict(
            tickvals=np.linspace(0, 1, 6),
            ticktext=[f'{int(x*100)}%' for x in np.linspace(0, 1, 6)],
            title="Desconto (%)"
        ),
        yaxis=dict(
            title="Frequência"
        ),
        bargap=0.05,
        template='plotly_white'
    )

    return fig


def grafico_dispersao(df):
    fig = px.scatter(
        df,
        x='N_Avaliações',
        y='Qtd_Vendidos_Cod',
        color='Nota',
        size='Desconto',
        color_continuous_scale='viridis',
        title="Relação entre Avaliações e Quantidade Vendida"
    )
    fig.update_layout(template='plotly_white')
    return fig


def grafico_mapa_calor(df):
    corr = df.drop(columns=['Unnamed: 0'], errors='ignore').corr(numeric_only=True)
    corr = corr.round(2)

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='RdBu',
        zmid=0,
        showscale=True,
        annotation_text=corr.values,
        hoverinfo="z"
    )

    fig.update_layout(
        title="Mapa de Calor - Correlação Entre os Dados",
        xaxis_nticks=36,
        width=1300,
        height=600,
        margin=dict(l=150, r=20, t=80, b=150)
    )
    return fig


def grafico_barras_temporada(df, temporada):
    df_grouped = df.groupby(['Temporada', 'Título'])['Qtd_Vendidos_Cod'].sum().reset_index()
    if temporada != "todas":
        df_grouped = df_grouped[df_grouped['Temporada'] == temporada]

    top_5 = df_grouped.groupby('Temporada').apply(lambda x: x.nlargest(5, 'Qtd_Vendidos_Cod')).reset_index(drop=True)

    fig = px.bar(
        top_5,
        x='Título',
        y='Qtd_Vendidos_Cod',
        color='Temporada',
        title=f"Top 5 Produtos Mais Vendidos ({'todas as temporadas' if temporada == 'todas' else temporada})",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(template='plotly_white', xaxis_title="Produto", yaxis_title="Qtd Vendidos (codificado)")
    return fig


def grafico_pizza(df):
    genero_counts = df["Gênero"].value_counts()
    limite = genero_counts.sum() * 0.02
    genero_ajustado = genero_counts.copy()
    outros = genero_ajustado[genero_ajustado < limite].sum()
    genero_ajustado = genero_ajustado[genero_ajustado >= limite]
    genero_ajustado["Outros"] = outros

    fig = px.pie(
        names=genero_ajustado.index,
        values=genero_ajustado.values,
        title="Distribuição de Produtos por Gênero",
        hole=0.4
    )

    # 🔥 Aqui faz as labels aparecerem no gráfico
    fig.update_traces(textinfo='label+percent')

    return fig


def grafico_densidade(df):
    # Colunas esperadas
    xcol = "Desconto_MinMax"
    ycol = "Nota_MinMax"

    # Verifica existência das colunas
    if xcol not in df.columns or ycol not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Colunas '{xcol}' ou '{ycol}' não encontradas no dataset.",
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="Densidade (dados ausentes)")
        return fig

    # Converte para numérico e remove NaNs
    df_local = df[[xcol, ycol]].copy()
    df_local[xcol] = pd.to_numeric(df_local[xcol], errors='coerce')
    df_local[ycol] = pd.to_numeric(df_local[ycol], errors='coerce')
    df_local = df_local.dropna()

    # Verifica se há dados suficientes
    if len(df_local) < 10:
        fig = go.Figure()
        msg = "Dados insuficientes para calcular densidade (precisa de pelo menos 10 pares válidos)."
        fig.add_annotation(text=msg, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="Densidade (poucos dados)")
        return fig

    # Tenta criar density_contour; se falhar, usa density_heatmap
    try:
        fig = px.density_contour(
            df_local,
            x=xcol,
            y=ycol,
            title="Relação entre Desconto (%) e Nota Média (normalizados)",
            color_continuous_scale="Blues"
        )
        fig.update_traces(contours_coloring="fill")
        fig.update_layout(template='plotly_white')
        # ajustar ticks X/Y para aparência em %/nota (se quiser)
        fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                         ticktext=[f"{int(v*100)}%" for v in [0,0.25,0.5,0.75,1.0]])
        fig.update_yaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                         ticktext=[f"{v:.1f}" for v in np.linspace(0,5,6)[::1][:5]])  # só exemplo
        return fig
    except Exception as e:
        # Fallback para density_heatmap (mais tolerante)
        try:
            fig = px.density_heatmap(
                df_local,
                x=xcol,
                y=ycol,
                nbinsx=30,
                nbinsy=30,
                color_continuous_scale="Blues",
                title="Relação entre Desconto (%) e Nota Média (heatmap fallback)"
            )
            fig.update_layout(template='plotly_white')
            fig.update_xaxes(tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                             ticktext=[f"{int(v*100)}%" for v in [0,0.25,0.5,0.75,1.0]])
            return fig
        except Exception as e2:
            # Retorna figura com mensagem de erro (não quebra o callback)
            fig = go.Figure()
            fig.add_annotation(text=f"Erro ao gerar densidade: {str(e)} / fallback: {str(e2)}",
                               xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title="Densidade - erro interno")
            return fig


def grafico_regressao(df):
    # Robustez: se statsmodels não existir, informa claramente
    xcol = "Desconto_MinMax"
    ycol = "Nota_MinMax"
    if xcol not in df.columns or ycol not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Colunas '{xcol}' ou '{ycol}' não encontradas.",
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="Regressão (dados ausentes)")
        return fig

    df_local = df[[xcol, ycol]].copy()
    df_local[xcol] = pd.to_numeric(df_local[xcol], errors='coerce')
    df_local[ycol] = pd.to_numeric(df_local[ycol], errors='coerce')
    df_local = df_local.dropna()

    if len(df_local) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Dados insuficientes para regressão.",
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="Regressão (poucos dados)")
        return fig

    # Tenta usar trendline (precisa de statsmodels)
    try:
        fig = px.scatter(
            df_local,
            x=xcol,
            y=ycol,
            trendline="ols",
            title="Tendência entre Desconto (norm) e Nota (norm)"
        )
        fig.update_layout(template='plotly_white')
        return fig
    except Exception as e:
        # Se statsmodels não estiver instalado, calcula linha manualmente com numpy
        try:
            X = df_local[xcol].values.reshape(-1, 1)
            y = df_local[ycol].values
            coef = np.polyfit(df_local[xcol].values, df_local[ycol].values, 1)
            trend_x = np.linspace(df_local[xcol].min(), df_local[xcol].max(), 100)
            trend_y = np.polyval(coef, trend_x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_local[xcol], y=df_local[ycol], mode='markers', name='Pontos'))
            fig.add_trace(go.Scatter(x=trend_x, y=trend_y, mode='lines', name='Tendência', line=dict(color='red')))
            fig.update_layout(title=f"Regressão (linha calculada). Coef: {coef[0]:.3f}, Intercepto: {coef[1]:.3f}",
                              template='plotly_white')
            return fig
        except Exception as e2:
            fig = go.Figure()
            fig.add_annotation(text=f"Erro ao gerar regressão: {str(e)} / fallback: {str(e2)}",
                               xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title="Regressão - erro interno")
            return fig


# ---------------------- #
#     DASH APP SETUP     #
# ---------------------- #

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Interativo - E-commerce", style={'textAlign': 'center'}),
    html.Hr(),

    dcc.Dropdown(
        id="grafico_selector",
        options=[
            {"label": "Histograma", "value": "hist"},
            {"label": "Dispersão", "value": "disp"},
            {"label": "Mapa de Calor", "value": "heat"},
            {"label": "Top 5 Produtos por Temporada", "value": "bar"},
            {"label": "Pizza (Gênero)", "value": "pie"},
            {"label": "Densidade", "value": "dens"},
            {"label": "Regressão", "value": "reg"}
        ],
        value="hist",
        clearable=False,
        style={'width': '50%', 'margin': '0 auto'}
    ),

    html.Br(),

    # Dropdown de temporada (inicialmente oculto)
    html.Div(
        id="temporada_div",
        children=[
            html.Label("Selecione uma temporada:"),
            dcc.Dropdown(
                id="temporada_selector",
                options=[
                    {"label": "Todas", "value": "todas"},
                    {"label": "Primavera/Verão", "value": "primavera/verão"},
                    {"label": "Outono/Inverno", "value": "outono/inverno"},
                    {"label": "Anual", "value": "anual"}
                ],
                value="todas",
                clearable=False
            ),
        ],
        style={'width': '40%', 'margin': '0 auto', 'display': 'none'}
    ),

    html.Br(),
    dcc.Graph(id="grafico_display")
])


# ---------------------- #
#     CALLBACKS DASH     #
# ---------------------- #

@app.callback(
    Output("temporada_div", "style"),
    Input("grafico_selector", "value")
)
def toggle_temporada_selector(grafico):
    """Mostra o dropdown apenas no gráfico de barras"""
    if grafico == "bar":
        return {'width': '40%', 'margin': '0 auto', 'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output("grafico_display", "figure"),
    Input("grafico_selector", "value"),
    Input("temporada_selector", "value")
)
def update_graph(grafico, temporada):
    if grafico == "hist":
        return grafico_histograma(df)
    elif grafico == "disp":
        return grafico_dispersao(df)
    elif grafico == "heat":
        return grafico_mapa_calor(df)
    elif grafico == "bar":
        return grafico_barras_temporada(df, temporada)
    elif grafico == "pie":
        return grafico_pizza(df)
    elif grafico == "dens":
        return grafico_densidade(df)
    elif grafico == "reg":
        return grafico_regressao(df)
    return {}


# ---------------------- #
#     EXECUTAR APP       #
# ---------------------- #

if __name__ == "__main__":
    app.run(debug=True)
