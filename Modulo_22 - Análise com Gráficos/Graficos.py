import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Título, Nota, N_Avaliações, Desconto, Marca, Material, Gênero, Temporada, Review1,
# Review2, Review3, Qtd_Vendidos, Preço, Nota_MinMax, N_Avaliações_MinMax,
# Desconto_MinMax, Preço_MinMax, Marca_Cod, Material_Cod, Temporada_Cod,
# Qtd_Vendidos_Cod, Marca_Freq, Material_Freq

df = pd.read_csv("ecommerce_estatistica.csv")
print(df.head(10).to_string(), "\n\n")



# --- Gráfico de Histograma  ---#
plt.figure(figsize=(10, 6))

# Histograma do desconto normalizado
sns.histplot(
    df['Desconto_MinMax'],
    bins=20,
    kde=True,                     # adiciona a linha de densidade
    color='royalblue',
    alpha=0.7
)

# Ajuste do eixo x para mostrar em porcentagem real (0 a 100%)
plt.xticks(
    ticks=np.linspace(0, 1, 6),
    labels=[f'{int(x*100)}%' for x in np.linspace(0, 1, 6)]
)

plt.title('Gráfico de Histograma\nDistribuição dos Descontos (%)', fontsize=14)
plt.xlabel('Desconto (%)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

print("Insight do Gráfico de Histograma:\n"
      "De acordo com a alta concentração perto de 0-10%, pode-se perceber que a maioria "
      "dos produtos possui pouco ou nenhum desconto, indicando promoções pontuais ou "
      "limitadas a uma pequena parte do catálogo.\n"
      "A queda progressiva, conforme a porcentagem aumenta, indica que grandes descontos "
      "provavelmente são usados em liquidações, fim de coleções ou campanhas específicas.")



# --- Gráfico de dispersão --- #

    # Qtd_Vendidos_Cod/ -N_Avaliações; N_Avaliações_MinMax = 0.91

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x='N_Avaliações',
    y='Qtd_Vendidos_Cod',
    hue='Nota',
    size='Desconto',
    palette='viridis',
    alpha=0.7
)

plt.title('Gráfico de Dispersão\nRelação entre Avaliações e Quantidade Vendida por Nota e Desconto', fontsize=13)
plt.xlabel('Número de Avaliações')
plt.ylabel('Quantidade Vendida (Codificada)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Nota')
plt.tight_layout()
plt.show()

print("\nInsight do Gráfico de Dispersão - Relação entre Avaliações e Quantidade Vendida por Nota e Desconto:\n"
      "Há uma correlação positiva entre o número de avaliações e a quantidade vendida, "
      "o que sugere que produtos com mais engajamento tendem a gerar mais vendas  "
      "(possivelmente por efeito social; confiança baseada em avaliações). "
      "Além disso, observa-se que produtos com descontos moderados e notas altas estão "
      "concentrados nas faixas de vendas mais elevadas, reforçando a ideia de que promoções "
      "bem calibradas e boa reputação formam a combinação ideal para performance comercial.")


# Mapa de calor
# --- Análise de Correlação dos Dados --- #
corr = df.drop(columns=['Unnamed: 0']).corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 8})
plt.title('Mapa de Calor\nCorrelação Entre os Dados', fontsize=14)
plt.xticks(rotation=60)  # rotaciona os rótulos do eixo x em 60°
plt.tight_layout()
plt.show()

    # N_Avaliações/Qtd_Vendidos_Cod = 0.91
    # N_Avaliações_MinMax/Qtd_Vendidos_Cod = 0.91
    # Qtd_Vendidos_Cod/ -N_Avaliações; N_Avaliações_MinMax = 0.91
    # Material_Freq/Material_Cod = -0.52

print("\nInsight do Mapa de Calor - Correlação Entre os Dados:\n"
      "“O mapa de calor revela que o 'Desconto' tem correlação positiva "
      "com 'Quantidade de Vendidos', sugerindo que promoções aumentam o volume de vendas. "
      "No entanto, a correlação entre 'Desconto' e 'Nota' é apenas moderada, "
      "indicando que descontos ajudam, mas a satisfação do cliente também depende de "
      "outros fatores, como qualidade e expectativa do produto.")



# --- Gráfico de Barra --- #
# Mantém apenas as duas temporadas principais, o restante vira 'anual'
df['Temporada'] = df['Temporada'].apply(
    lambda x: x if x in ['primavera/verão', 'outono/inverno']
    else 'anual'
)
#print("\n", df.drop(columns='Unnamed: 0').head(10).to_string(), "\n\n\n")

# Agrupar os dados por 'Temporada' e 'Título', calculando a soma da quantidade de produtos vendidos
df_grouped = df.groupby(['Temporada', 'Título'])['Qtd_Vendidos_Cod'].sum().reset_index()

# Filtrar os top 5 produtos mais vendidos para cada temporada
top_5_per_season = df_grouped.groupby('Temporada').apply(lambda x: x.nlargest(5, 'Qtd_Vendidos_Cod')).reset_index(drop=True)

# Plotando os dados filtrados
plt.figure(figsize=(14, 6))
sns.barplot(x='Temporada', y='Qtd_Vendidos_Cod', hue='Título', data=top_5_per_season, palette='tab20')

plt.xticks(rotation=0)
plt.title('Gráfico de Barras\nTop 5 Produtos Mais Vendidos por Temporada', fontsize=14)
plt.xlabel('Temporada', fontsize=12)
plt.ylabel('Qtd Vendidos (codificado)', fontsize=12)
plt.legend(title='Produto', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("\nInsight do Gráfico de Barras - Top 5 Produtos Mais Vendidos por Temporada:\n"
      "É possível perceber uma sazonalidade clara: os 5 produtos mais vendidos em cada "
      "temporada não se repetem em outros momentos do ano. Em contrapartida, o item "
      "Cueca Box(Boxer) aparece em todo o ano. Pode ser um item interessante de focar "
      "e aproveitar a boa saída."
      "\nO padrão de roupas leves no verão e pesadas no inverno indicam a necessidade "
      "de ajustar estoque e investimentos conforme a temporada, evitando excesso de "
      "inventário fora de época.")



# --- Gráfico de pizza --- #

genero_counts = df["Gênero"].value_counts()  # Conta a frequência de cada gênero

# Junta categorias com menos de 2% em "Outros" (melhora muito a leitura)
limite = genero_counts.sum() * 0.02
genero_ajustado = genero_counts.copy()
outros = genero_ajustado[genero_ajustado < limite].sum()
genero_ajustado = genero_ajustado[genero_ajustado >= limite]
genero_ajustado["Outros"] = outros

# Cria o gráfico
plt.figure(figsize=(8, 8))
plt.pie(
    genero_ajustado,
    labels=genero_ajustado.index,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85,             # afasta o texto da borda
    labeldistance=1.05,           # afasta as labels do centro
    wedgeprops={'edgecolor': 'white'}
)
plt.title("Gráfico de Pizza\nDistribuição de Produtos por Gênero")
plt.tight_layout()
plt.show()

print("\nInsight do Gráfico de Pizza - Distribuição de Produtos por Gênero:\n"
      "Os produtos se mostram bem equilibrados por gênero, com uma leve tendência "
      "de maior foco em produtos masculinos. Pode ser interessante pensar em estratégias "
      "para atrair mais o público feminino.")



# --- Gráfico de densidade --- #
# Configuração do gráfico
plt.figure(figsize=(8,6))

# Gráfico de densidade bivariada (2D)
sns.kdeplot(
    data=df,
    x="Desconto_MinMax",
    y="Nota_MinMax",
    fill=True,
    cmap="Blues",
    thresh=0.05,
    levels=50
)

# Conversão dos eixos para valores reais

# Eixo X: mostrar os descontos reais (0% a 100%)
# Calcula os valores reais com base no intervalo real de "Desconto"
min_desc, max_desc = df["Desconto"].min(), df["Desconto"].max()
xticks_norm = [0, 0.25, 0.5, 0.75, 1.0]
xticks_real = [round(min_desc + (max_desc - min_desc) * x, 1) for x in xticks_norm]
plt.xticks(xticks_norm, [f"{x:.0f}%" for x in xticks_real])

# Eixo Y: mostrar as notas reais (0 a 5)
yticks_norm = [0, 0.25, 0.5, 0.75, 1.0]
yticks_real = [round(0 + (5 - 0) * y, 1) for y in yticks_norm]
plt.yticks(yticks_norm, yticks_real)

# Rótulos e título
plt.title("Gráfico de Densidade\nRelação entre Desconto (%) e Nota Média (0–5)", fontsize=14)
plt.xlabel("Desconto Real (%)", fontsize=12)
plt.ylabel("Nota Média", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)

plt.show()

print("\nInsight do Gráfico de Densidade - Relação entre Desconto (%) e Nota Média (0-5):\nOs clientes parecem mais satisfeitos com produtos que oferecem descontos "
      "moderados (em torno de 5% a 20%), indicando que promoções equilibradas "
      "aumentam a percepção de valor sem comprometer a qualidade percebida.")



# --- Gráfico de Regressão --- #
# Configuração do gráfico
plt.figure(figsize=(8,6))

# Gráfico de regressão (tendência)
sns.regplot(
    data=df,
    x="Desconto_MinMax",    # eixo X normalizado
    y="Nota_MinMax",        # eixo Y normalizado
    scatter_kws={"alpha": 0.5, "s": 40},  # transparência e tamanho dos pontos
    line_kws={"color": "red", "linewidth": 2},  # estilo da linha de tendência
)

# Ajustar eixos para mostrar valores reais

# Eixo X → Desconto real (%)
min_desc, max_desc = df["Desconto"].min(), df["Desconto"].max()
xticks_norm = [0, 0.25, 0.5, 0.75, 1.0]
xticks_real = [round(min_desc + (max_desc - min_desc) * x, 1) for x in xticks_norm]
plt.xticks(xticks_norm, [f"{x:.0f}%" for x in xticks_real])

# Eixo Y → Nota real (0–5)
yticks_norm = [0, 0.25, 0.5, 0.75, 1.0]
yticks_real = [round(0 + (5 - 0) * y, 1) for y in yticks_norm]
plt.yticks(yticks_norm, yticks_real)

# Rótulos e estilo
plt.title("Gráfico de  Regressão\nTendência entre Desconto (%) e Nota Média (0–5)", fontsize=14)
plt.xlabel("Desconto Real (%)", fontsize=12)
plt.ylabel("Nota Média", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)

plt.show()

print("\nInsight do Gráfico de Regressão - Tendência entre Desconto (%) e Nota Média (0–5):\n"
      "Produtos com descontos equilibrados tendem a manter a satisfação dos clientes, "
      "enquanto reduções muito agressivas podem estar associadas a itens de menor "
      "qualidade ou de liquidação.")

