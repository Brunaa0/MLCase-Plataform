########### Bibliotecas Necessárias ###########


# -------------------------------------
# Bibliotecas do Streamlit
# -------------------------------------
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------
# Manipulação e Análise de Dados
# -------------------------------------
import pandas as pd
import numpy as np

# -------------------------------------
# Visualização de Dados
# -------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------------------
# Modelos de Machine Learning
# -------------------------------------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree, neighbors

# -------------------------------------
# Seleção de Features
# -------------------------------------
from mlxtend.feature_selection import SequentialFeatureSelector

# -------------------------------------
# Métricas de Avaliação
# -------------------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# -------------------------------------
# Pré-Processamento e Pipeline
# -------------------------------------
from sklearn.model_selection import (
    train_test_split, KFold, LeaveOneOut, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# -------------------------------------
# Utilitários
# -------------------------------------
import os
import joblib
import pickle
import io
from io import BytesIO
import tempfile
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from scipy.sparse import csr_matrix
import scipy
import time
import json
import requests
import unidecode

# -------------------------------------
# Bibliotecas Adicionais para Geração de Relatórios
# -------------------------------------
from fpdf import FPDF
import io
import tempfile
import requests
from datetime import datetime
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import inch


##############################################

# Função JavaScript para a página voltar ao topo
scroll_to_top_js = """
<script>
    function scrollToTop() {
        window.scrollTo(0, 0);
    }
</script>
"""

# Adiciona o JavaScript na página
components.html(scroll_to_top_js, height=0, width=0)  # Define altura e largura para manter invisível

# Ajuste das opções de exibição do Pandas Styler
pd.set_option("styler.render.max_elements", 2000000)  # Ajuste conforme necessário
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

##############################################
def fix_dataframe_types(df):
    """Corrigir tipos de dados em um DataFrame para compatibilidade com PyArrow"""
    # Verificar se é um objeto Styler e extrair o DataFrame
    if hasattr(df, 'data'):  # Styler objects have a .data attribute
        df = df.data
    elif hasattr(df, 'render') and not hasattr(df, 'copy'):  # Another way to detect Styler
        # Para versões mais recentes do pandas
        if hasattr(df, '_data'):
            df = df._data
        # Para versões bem recentes do pandas onde pode ser diferente
        elif hasattr(df, 'data'):
            df = df.data
        # Se ainda não conseguiu extrair o DataFrame
        else:
            # Tentar converter para dict primeiro e depois para DataFrame
            try:
                df = pd.DataFrame(df.to_dict())
            except:
                # Se tudo falhar, retornar um DataFrame vazio
                return pd.DataFrame()
    
    # Se não for DataFrame, retornar vazio
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
        
    # Criar uma cópia para não modificar o original
    df_fixed = df.copy()
    
    # Converter colunas problemáticas
    for col in df_fixed.columns:
        # Converter Int64 para int64 padrão
        if hasattr(df_fixed[col], 'dtype') and str(df_fixed[col].dtype) == 'Int64':
            df_fixed[col] = df_fixed[col].fillna(-1).astype('int64')
        
        # Converter objetos complexos para string
        elif df_fixed[col].dtype == 'object':
            try:
                # Tentar converter para string
                df_fixed[col] = df_fixed[col].astype(str)
            except:
                # Se falhar, aplicar uma conversão manual
                df_fixed[col] = df_fixed[col].apply(lambda x: str(x) if x is not None else "")
    
    return df_fixed

##############################################
# Função para configurar a sidebar fixa
def configure_sidebar():
    with st.sidebar:
        st.image(
            "https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg",
            width=80,  # Define o tamanho da imagem diretamente
            caption="Logótipo da Escola"
        )
        st.markdown("<p>MLCase - Plataforma de Machine Learning</p>", unsafe_allow_html=True)
        st.markdown("<p><b>Autora:</b> Bruna Sousa</p>", unsafe_allow_html=True)


# Configurar a sidebar
configure_sidebar()

##############################################

# FUNÇÃO DE UPLOAD 

# Função para inicializar variáveis de estado
def initialize_state():
    st.session_state.step = 'data_preview'
    st.session_state.selected_columns = []
    st.session_state.numeric_types = {}
    st.session_state.variable_types = {}
    st.session_state.treatment_state = {}
    st.session_state.all_treated = False

# Função auxiliar para escolher o delimitador para ficheiros CSV
def choose_delimiter():
    # Lista de delimitadores possíveis, incluindo a opção em branco
    delimiters = [",", ";", "\t", "|", "Outro"]  # Adiciona a opção "Outro"
    
    # Cria o selectbox para o usuário escolher o delimitador
    delimiter = st.sidebar.selectbox("Escolha o delimitador para CSV", delimiters, index=0)
    
    # Se o usuário escolher a opção "Outro", permite que ele insira um delimitador personalizado
    if delimiter == "Outro":
        delimiter = st.sidebar.text_input("Digite o delimitador personalizado:")
    
    return delimiter


# Função para a etapa de upload do arquivo
def upload_file():
    st.title("MLCase - Plataforma de Machine Learning")

    # Seleção de tipo de arquivo e definição de delimitador padrão
    file_type = st.sidebar.selectbox("Selecione o tipo de arquivo", ["CSV", "Excel", "JSON"])
    delimiter = ","  # Padrão para CSV

    # Upload de arquivo e escolha do delimitador, se CSV
    if file_type == "CSV":
        delimiter = choose_delimiter()
        file = st.sidebar.file_uploader("Carregar arquivo", type=["csv"])
    elif file_type == "Excel":
        file = st.sidebar.file_uploader("Carregar arquivo", type=["xlsx", "xls"])
    elif file_type == "JSON":
        file = st.sidebar.file_uploader("Carregar arquivo", type=["json"])

    # Carrega o arquivo, se fornecido, e configura o estado
    if file is not None:
        try:
            st.session_state.data = load_data(file_type, file, delimiter)
            initialize_state()
            st.sidebar.success(f"Conjunto de dados {file_type} carregado com sucesso!")

            # Botão para avançar para a pré-visualização dos dados
            if st.sidebar.button("Dados Carregados"):
                st.session_state.step = 'data_preview'
                st.stop()  # Atualiza a página para refletir o novo estado

        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {e}")

# Função para carregar dados com cache
@st.cache_data
def load_data(file_type, file, delimiter):
    if file_type == "CSV":
        return pd.read_csv(file, delimiter=delimiter)
    elif file_type == "Excel":
        return pd.read_excel(file)
    elif file_type == "JSON":
        return pd.read_json(file)

##############################################
# FUNÇÃO DE SELEÇÃO DE COLUNAS 

# Função para visualização de dados e seleção de colunas e tipos de dados
def data_preview():
    st.subheader("Pré-visualização dos dados")
    st.dataframe(fix_dataframe_types(st.session_state.data.head()))

    # Seleção de colunas
    columns = st.session_state.data.columns.tolist()
    selected_columns = st.multiselect("Colunas", columns, columns)
    st.session_state.selected_columns = selected_columns

    # Preservar transformações no estado global
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = st.session_state.data.copy()
    else:
        # Filtra apenas colunas selecionadas sem perder transformações
        st.session_state.filtered_data = st.session_state.data[selected_columns]


    if selected_columns:
        st.subheader("Identificar tipos de variáveis")
        if 'variable_types' not in st.session_state:
            st.session_state.variable_types = {}
        variable_types = st.session_state.variable_types
        st.session_state.numeric_types = {}

        # Definir tipos de variáveis e configurar numéricos
        for col in selected_columns:
            var_type = st.selectbox(
                f"Tipo de variável para {col}",
                ["Numérica", "Categórica", "Data"],
                index=0 if pd.api.types.is_numeric_dtype(st.session_state.filtered_data[col]) else 1,
                key=f"var_{col}"
            )
            variable_types[col] = var_type

            # Configurações de tipos para variáveis numéricas
            if var_type == "Numérica":
                num_type = st.selectbox(
                    f"Tipo numérico para {col}",
                    ["Int", "Float", "Complex", "Dec", "Frac", "Bool"],
                    index=0 if pd.api.types.is_integer_dtype(st.session_state.filtered_data[col]) else 1,
                    key=f"num_{col}"
                )
                st.session_state.numeric_types[col] = num_type

                # Discretização - verifica antes se já foi aplicada
                if col not in st.session_state.filtered_data.columns or pd.api.types.is_numeric_dtype(st.session_state.filtered_data[col]):
                    if st.checkbox(f"Discretizar {col}?", key=f"discretize_{col}"):
                        discretize_column(col)
                else:
                    st.write(f"Coluna {col} já foi discretizada.")

        st.session_state.variable_types = variable_types

    # Atualizar estado global após processamento
    st.session_state.filtered_data = st.session_state.filtered_data.copy()

    # Navegação entre etapas
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Voltar"):
            # Apagar estados salvos explicitamente
            keys_to_reset = [
                'filtered_data', 'selected_columns', 'variable_types',
                'numeric_types', 'treatment_state'
            ]
            for key in keys_to_reset:
                st.session_state.pop(key, None)  # Remove se existir

            # Resetar para o dataset original
            st.session_state.data = st.session_state.data.copy()

            # Voltar para a etapa anterior
            st.session_state.step = 'file_upload'
            st.rerun()

    with col2:
        if st.button("Próxima etapa"):
            apply_numeric_types()
            st.session_state.step = 'missing_values'
            st.rerun()


# Função para aplicar tipos numéricos às colunas filtradas
def apply_numeric_types():
    for col, num_type in st.session_state.numeric_types.items():
        if col in st.session_state.filtered_data.columns:
            st.session_state.filtered_data[col] = convert_numeric_type(st.session_state.filtered_data[col], num_type)

# Conversão de tipos de dados numéricos
def convert_numeric_type(series, num_type):
    try:
        if num_type == "Int":
            return pd.to_numeric(series, errors='coerce').astype('Int64')
        elif num_type == "Float":
            return pd.to_numeric(series, errors='coerce').astype(float)
        elif num_type == "Complex":
            return pd.to_numeric(series, errors='coerce').apply(lambda x: complex(x) if pd.notnull(x) else np.nan)
        elif num_type == "Dec":
            return series.apply(lambda x: Decimal(x) if pd.notnull(x) else np.nan)
        elif num_type == "Frac":
            return series.apply(lambda x: Fraction(x) if pd.notnull(x) else np.nan)
        elif num_type == "Bool":
            return series.apply(lambda x: str(x).strip().lower() in ['true', '1'])
        elif num_type == "Date":
            return pd.to_datetime(series, errors='coerce')
        elif num_type == "Duration":
            return pd.to_timedelta(series, errors='coerce')
        else:
            return series
    except Exception as e:
        st.error(f"Erro ao converter coluna {series.name} para tipo {num_type}: {e}")

# Função para discretizar uma coluna numérica
def discretize_column(col):
    # Botão de ajuda
    with st.expander("Como preencher os bins e labels?"):
        st.write("**Bins:** Intervalos numéricos para discretização.")
        st.write("**Labels:** Nomeiam os intervalos.")
        st.write("**Exemplo:**")
        st.write("- Bins: -2,1,2,6,inf")
        st.write("- Labels: Baixo, Médio, Alto, Muito Alto")

    # Diagnóstico antes de confirmar
    st.write("### Diagnóstico antes da discretização:")
    st.write(f"- **Mínimo:** {st.session_state.filtered_data[col].min()}")
    st.write(f"- **Máximo:** {st.session_state.filtered_data[col].max()}")
    st.write(f"- **Média:** {st.session_state.filtered_data[col].mean():.2f}")
    st.write(f"- **Mediana:** {st.session_state.filtered_data[col].median():.2f}")
    st.write(f"- **Valores ausentes antes:** {st.session_state.filtered_data[col].isna().sum()}")

    # Pré-preencher com exemplos
    bins_input = st.text_input(
        f"Digite os bins para {col} (separados por vírgulas)",
        value="-2,1,2,6,inf", key=f"bins_{col}"
    )
    labels_input = st.text_input(
        f"Digite os labels para {col} (separados por vírgulas)",
        value="Baixo,Médio,Alto,Muito Alto", key=f"labels_{col}"
    )

    # Botão para confirmar discretização
    if st.button(f"Confirmar Discretização para {col}", key=f"confirm_{col}"):
        if bins_input and labels_input:
            try:
                # Converter inputs
                bins = list(map(float, bins_input.split(',')))
                labels = labels_input.split(',')

                # Validar bins e labels
                if len(labels) != len(bins) - 1:
                    st.error(f"O número de labels deve ser igual ao número de bins menos um para a coluna {col}.")
                else:
                    # Garantir tipo float
                    st.session_state.filtered_data[col] = pd.to_numeric(
                        st.session_state.filtered_data[col], errors='coerce'
                    )

                    # Preencher valores faltantes com a mediana
                    median_value = st.session_state.filtered_data[col].median()
                    st.session_state.filtered_data[col].fillna(median_value, inplace=True)

                    # Diagnóstico após preenchimento
                    st.write(f"Valores ausentes após preenchimento: {st.session_state.filtered_data[col].isna().sum()}")

                    # Discretizar
                    categorized = pd.cut(
                        st.session_state.filtered_data[col],
                        bins=bins,
                        labels=labels,
                        include_lowest=True
                    )

                    # Garantir categórico e adicionar categoria para valores fora do intervalo
                    categorized = categorized.astype('category')
                    categorized = categorized.cat.add_categories(["Fora do Intervalo"])
                    categorized = categorized.fillna("Fora do Intervalo")

                    # Salvar no estado global e garantir consistência
                    st.session_state.filtered_data[col] = categorized
                    st.session_state.filtered_data = st.session_state.filtered_data.copy()

                    # Diagnóstico após salvar
                    st.success(f"Coluna {col} discretizada com sucesso!")
                    st.write(st.session_state.filtered_data[col].dtype)
                    st.write(st.session_state.filtered_data[col].unique())
                    st.write("Pré-visualização dos dados após discretização:")
                    st.dataframe(fix_dataframe_types(st.session_state.filtered_data.head()))

            except ValueError as e:
                st.error(f"Erro ao discretizar {col}: {e}")


##############################################
# FUNÇÃO DE TRATAMENTO DE VALORES OMISSOS

# Função para DataFrame com destaque para valores ausentes
def highlight_missing():
    def highlight_na(s):
        return ['background-color: yellow' if pd.isnull(v) else '' for v in s]
    return st.session_state.filtered_data.style.apply(highlight_na, subset=st.session_state.filtered_data.columns)

# Função para formatar valores na tabela
def format_table():
    formatted_df = st.session_state.filtered_data.copy()
    for col in formatted_df.columns:
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else 'NaN')
    return formatted_df

# Função para mostrar a pré-visualização com tipos de variáveis
def show_preview_with_types(variable_types):
    st.subheader("Pré-visualização dos dados com tipos de variáveis")
    st.write("Tipos de variáveis:")
    st.write(variable_types)
    
    # Usa o filtered_data diretamente
    formatted_df = format_table()
    st.dataframe(fix_dataframe_types(highlight_missing(formatted_df)))

# Função para aplicar tratamento de valores ausentes
def apply_missing_value_treatment(column, method, constant_value=None):
    # Usa diretamente o filtered_data do estado global
    data = st.session_state.filtered_data
    
    if pd.api.types.is_numeric_dtype(data[column]):
        if method == "Média":
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == "Mediana":
            data[column].fillna(data[column].median(), inplace=True)
        elif method == "Moda":
            data[column].fillna(data[column].mode().iloc[0], inplace=True)
        elif method == "Excluir":
            data.dropna(subset=[column], inplace=True)
        elif method == "Valor constante" and constant_value is not None:
            data[column].fillna(constant_value, inplace=True)
    else:
        if method == "Substituir por moda":
            data[column].fillna(data[column].mode().iloc[0], inplace=True)
        elif method == "Substituir por valor constante" and constant_value is not None:
            data[column].fillna(constant_value, inplace=True)
        elif method == "Manter valores ausentes":
            pass  # Não faz nada
        elif method == "Excluir":
            data.dropna(subset=[column], inplace=True)

    # Atualiza os dados processados no estado global
    st.session_state.filtered_data = data

def auto_select_method(column_name):
    # Usa diretamente o filtered_data
    column = st.session_state.filtered_data[column_name]
    missing_percentage = column.isnull().sum() / len(column)

    # Para colunas numéricas
    if pd.api.types.is_numeric_dtype(column):
        if missing_percentage > 0.5:
            return "Excluir"
        else:
            return "Substituir por Mediana"
    # Para colunas categóricas
    else:
        if missing_percentage > 0.5:
            return "Excluir"
        else:
            return "Substituir por Moda"

def display_missing_values(dataframe):
    # Verifica valores ausentes
    missing_data = dataframe.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    missing_data = missing_data.reset_index()
    missing_data.columns = ['Coluna', 'Valores Ausentes']

    if not missing_data.empty:
        st.write("Tabela de valores ausentes:")
        st.dataframe(fix_dataframe_types(missing_data))
    else:
        st.write("Não há valores ausentes.")

# Função para mostrar e tratar valores ausentes
def handle_missing_values():
    st.subheader("Tratamento de Valores Ausentes")

    # Acesso aos dados filtrados no estado da sessão
    filtered_data = st.session_state.get('filtered_data', None)

    if filtered_data is not None and not filtered_data.empty:
        # Exibir valores ausentes
        def display_missing_values(df):
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if not missing_data.empty:
                st.write("Resumo dos Valores Ausentes:")
                st.dataframe(fix_dataframe_types(missing_data.rename("Total de Valores Ausentes")))
            else:
                st.success("Não há valores ausentes nos dados.")

        # Exibir os valores ausentes
        display_missing_values(filtered_data)

        # Verificar se existem valores ausentes
        has_missing_values = filtered_data.isnull().any().any()

        if has_missing_values:
            if 'treatment_state' not in st.session_state:
                st.session_state.treatment_state = {
                    col: {"method": None, "constant": None}
                    for col in filtered_data.columns
                }

            # Exibir opções para cada coluna com valores ausentes
            for col in filtered_data.columns:
                if filtered_data[col].isnull().sum() > 0:
                    col_state = st.session_state.treatment_state.get(col, {"method": None, "constant": None})
                    is_numeric = pd.api.types.is_numeric_dtype(filtered_data[col])

                    if is_numeric:
                        options = ["Substituir por Média", "Substituir por Mediana", "Substituir por Moda", 
                                   "Substituir por Valor Constante", "Excluir", "Manter Valores Ausentes"]
                        missing_value_method = st.selectbox(
                            f"Método para tratar valores ausentes em {col}",
                            options,
                            index=options.index(col_state["method"]) if col_state["method"] in options else 0,
                            key=f"missing_value_{col}"
                        )
                        constant_value = None
                        if missing_value_method == "Substituir por Valor Constante":
                            constant_value = st.text_input(
                                f"Digite o valor constante para {col}:",
                                value=col_state["constant"] if col_state["constant"] else '',
                                key=f"constant_{col}"
                            )
                    else:
                        options = ["Substituir por Moda", "Substituir por Valor Constante", "Manter Valores Ausentes", "Excluir"]
                        missing_value_method = st.selectbox(
                            f"Método para tratar valores ausentes em {col}",
                            options,
                            index=options.index(col_state["method"]) if col_state["method"] in options else 0,
                            key=f"cat_missing_value_{col}"
                        )
                        constant_value = None
                        if missing_value_method == "Substituir por Valor Constante":
                            constant_value = st.text_input(
                                f"Digite o valor constante para {col}:",
                                value=col_state["constant"] if col_state["constant"] else '',
                                key=f"cat_constant_{col}"
                            )

                    # Atualizar o estado com as escolhas do usuário
                    st.session_state.treatment_state[col] = {"method": missing_value_method, "constant": constant_value}

            # Botão para aplicar os tratamentos
            if st.button("Aplicar tratamentos"):
                for col, treatment in st.session_state.treatment_state.items():
                    method = treatment["method"]
                    constant_value = treatment["constant"]

                    if method == "Substituir por Média":
                        filtered_data[col].fillna(filtered_data[col].mean(), inplace=True)
                    elif method == "Substituir por Mediana":
                        filtered_data[col].fillna(filtered_data[col].median(), inplace=True)
                    elif method == "Substituir por Moda":
                        filtered_data[col].fillna(filtered_data[col].mode().iloc[0], inplace=True)
                    elif method == "Substituir por Valor Constante" and constant_value is not None:
                        filtered_data[col].fillna(constant_value, inplace=True)
                    elif method == "Excluir":
                        filtered_data.dropna(subset=[col], inplace=True)

                st.session_state.data = filtered_data.copy()
                st.success("Tratamentos aplicados com sucesso!")


        # Navegação
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Voltar"):
                st.session_state.step = 'data_preview'
                st.rerun()
        with col2:
            if st.button("Próxima etapa"):
                st.session_state.step = 'outlier_detection'
                st.rerun()
    else:
        st.error("Nenhum dado disponível para tratamento de valores ausentes.")

##############################################
# FUNÇÃO DE TRATAMENTO DE OUTLIERS

# Função para detectar e calcular informações de outliers
@st.cache_data

def calculate_outliers(columns, data):
    variables_with_outliers = []
    outlier_summary = []

    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identificar outliers
            num_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
            if num_outliers > 0:
                percentage_outliers = (num_outliers / len(data[col])) * 100
                variables_with_outliers.append(col)
                outlier_summary.append({
                    "Variável": col,
                    "Total de Outliers": num_outliers,
                    "Percentagem de Outliers (%)": round(percentage_outliers, 2)
                })

    return variables_with_outliers, outlier_summary

# Interface de detecção e tratamento de outliers
def outlier_detection():
    st.subheader("Detecção de Outliers")

    # Armazenar os dados originais (apenas na primeira execução)
    if 'original_data' not in st.session_state:
        st.session_state.original_data = st.session_state.data.copy()

    # **Boxplot Inicial - Fixo**
    st.write("### Boxplot Inicial (Dados Originais)")
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.original_data.boxplot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # Inicializar estado global
    if 'treated_columns' not in st.session_state:
        st.session_state.treated_columns = []
    if 'outlier_details' not in st.session_state:
        st.session_state.outlier_details = {}
    if 'initial_limits' not in st.session_state:
        st.session_state.initial_limits = {}
    if 'columns_with_outliers' not in st.session_state:
        st.session_state.columns_with_outliers = []  # Apenas variáveis com outliers
    if 'outlier_treatment_state' not in st.session_state:
        st.session_state.outlier_treatment_state = {}
    if 'all_outliers_treated' not in st.session_state:  # Novo estado
        st.session_state.all_outliers_treated = False

    # Garantir que os dados estão disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("Os dados não estão carregados! Volte para a etapa anterior.")
        return

    # Identificar colunas numéricas
    numeric_columns = list(st.session_state.data.select_dtypes(include=[np.number]).columns)
    outlier_summary = []

    # Processar cada coluna para calcular limites e outliers
    for col in numeric_columns:
        # Ignorar colunas já tratadas
        if col in st.session_state.treated_columns:
            continue

        # Calcular limites
        Q1 = st.session_state.data[col].quantile(0.25)
        Q3 = st.session_state.data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Contar outliers
        total_outliers = len(st.session_state.data[(st.session_state.data[col] < lower_bound) | 
                                                   (st.session_state.data[col] > upper_bound)])
        total_severe_outliers = len(st.session_state.data[(st.session_state.data[col] < (Q1 - 3.0 * IQR)) | 
                                                           (st.session_state.data[col] > (Q3 + 3.0 * IQR))])

        # Se a coluna tiver outliers, salvar detalhes
        if total_outliers > 0:
            # Armazenar limites e detalhes no estado global
            st.session_state.initial_limits[col] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            st.session_state.outlier_details[col] = {
                "total_outliers": total_outliers,
                "total_severe_outliers": total_severe_outliers,
                "skewness": st.session_state.data[col].skew()
            }

            # Adicionar ao resumo
            outlier_summary.append({
                "Nome variável": col,
                "Total de outliers": total_outliers,
                "Total de outliers severos": total_severe_outliers
            })

            # Adicionar na lista de variáveis com outliers
            if col not in st.session_state.columns_with_outliers:
                st.session_state.columns_with_outliers.append(col)

    # Salvar resumo no estado
    st.session_state.initial_outlier_summary = outlier_summary

    # Verificar se ainda há outliers não tratados
    remaining_outliers = [col for col in st.session_state.columns_with_outliers 
                          if col not in st.session_state.treated_columns]

    # Caso não existam mais outliers para tratar
    if not remaining_outliers:
        # Se nunca houve outliers desde o início
        if not outlier_summary and not st.session_state.columns_with_outliers:
            st.success("Nenhum outlier detectado nas variáveis numéricas!")
        else:
            st.success("Todos os outliers detectados foram tratados!")  # Novo aviso
    else:
        # Mostrar resumo dos outliers restantes
        st.write("Resumo dos Outliers:")
        st.dataframe(fix_dataframe_types(pd.DataFrame(outlier_summary)))

    # **Exibir e tratar apenas variáveis com outliers não tratados**
    for col in remaining_outliers:  # Somente as variáveis pendentes
        # Diagnóstico
        st.write(f"**Diagnóstico para {col}:**")
        details = st.session_state.outlier_details[col]
        st.write(f"- Total: {len(st.session_state.data)}")
        st.write(f"- Outliers: {details['total_outliers']} ({(details['total_outliers'] / len(st.session_state.data)):.2%})")
        st.write(f"- Outliers Severos: {details['total_severe_outliers']} ({(details['total_severe_outliers'] / len(st.session_state.data)):.2%})")
        st.write(f"- Assimetria (Skewness): {details['skewness']:.2f}")

        # Sugestão automática
        if col not in st.session_state.outlier_treatment_state:
            suggested_method = auto_select_outlier_treatment(
                col, st.session_state.data, st.session_state.initial_limits[col]["lower_bound"], st.session_state.initial_limits[col]["upper_bound"]
            )
            st.session_state.outlier_treatment_state[col] = suggested_method

        # Selectbox com chave única
        method = st.selectbox(
            f"Selecione o método para tratar outliers em {col}",
            ["Sem Ação", "Remover Outliers", "Remover Outliers Severos", "Substituir por Limites", "Substituir por Média", "Substituir por Mediana"],
            index=["Sem Ação", "Remover Outliers", "Remover Outliers Severos", "Substituir por Limites", "Substituir por Média", "Substituir por Mediana"].index(
                st.session_state.outlier_treatment_state[col]
            ),
            key=f"outlier_method_{col}_{len(st.session_state.treated_columns)}"  # Chave única
        )

        # Botão para aplicar tratamento
        if st.button(f"Aplicar tratamento em {col}"):
            apply_outlier_treatment(col, method, st.session_state.initial_limits[col]["lower_bound"], st.session_state.initial_limits[col]["upper_bound"])
            
            # Esta linha é crucial - ela marca a coluna como tratada
            if col not in st.session_state.treated_columns:
                st.session_state.treated_columns.append(col)
                
            st.rerun()  # Atualizar a página após o tratamento

    # **Boxplot Final**
    st.write("### Boxplot Após Tratamento")
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.data.boxplot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # **Tabela para Verificar Outliers Restantes**
    st.write("### Resumo Final de Outliers")

    # Função para calcular outliers restantes
    def calculate_remaining_outliers(data, numeric_columns):
        outlier_summary = []
        for col in numeric_columns:
            # Se esta coluna foi tratada, não deve ter mais outliers
            if col in st.session_state.treated_columns:
                outlier_summary.append({
                    "Coluna": col,
                    "Outliers Restantes": 0,
                    "Percentagem (%)": 0.00
                })
                continue
                
            # Para colunas não tratadas, calcular normalmente
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
    
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            # Contar os outliers restantes
            num_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
            percentage_outliers = (num_outliers / len(data)) * 100
    
            outlier_summary.append({
                "Coluna": col,
                "Outliers Restantes": num_outliers,
                "Percentagem (%)": round(percentage_outliers, 2)
            })
        return pd.DataFrame(outlier_summary)

    # Calcular e exibir a tabela
    numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns
    remaining_outliers = calculate_remaining_outliers(st.session_state.data, numeric_columns)
    st.write(remaining_outliers)

    
    # **Botão para próxima etapa sempre visível**
    if st.button("Próxima etapa"):
        st.session_state.step = 'data_summary'
        st.rerun()

# Função de sugestão automática corrigida
def auto_select_outlier_treatment(col, data, lower_bound, upper_bound):
    """Função para sugerir tratamento de outliers com base nos dados"""
    # Proporção de outliers
    total = len(data)
    total_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
    total_severe_outliers = len(data[(data[col] < (lower_bound - 1.5 * (upper_bound - lower_bound))) |
                                     (data[col] > (upper_bound + 1.5 * (upper_bound - lower_bound)))])
    percentage = total_outliers / total
    severe_percentage = total_severe_outliers / total

    # Verificar simetria dos dados
    skewness = data[col].skew()

    # Regras baseadas na proporção de outliers
    if severe_percentage > 0.10:  # Mais de 10% são severos
        return "Remover Outliers Severos"
    elif percentage > 0.20:  # Mais de 20% são outliers
        return "Remover Outliers"
    elif percentage > 0.05:  # Entre 5% e 20%
        return "Substituir por Limites"
    else:
        # Escolha entre média e mediana com base na assimetria
        if abs(skewness) > 1:
            return "Substituir por Mediana"
        else:
            return "Substituir por Média"

def apply_outlier_treatment(col, method, lower_bound, upper_bound):
    """Aplica o tratamento de outliers na coluna especificada."""
    # Obter os dados do estado
    data = st.session_state.data
    
    if method == "Remover Outliers":
        # Remover todos os outliers (valores além de 1.5 * IQR)
        st.session_state.data = data[
            (data[col] >= lower_bound) & (data[col] <= upper_bound)
        ]
        st.success(f"Todos os outliers removidos na coluna '{col}'.")

    elif method == "Remover Outliers Severos":
        # Remover apenas outliers severos (>3xIQR)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definição correta de outliers severos (3 * IQR)
        severe_lower = Q1 - 3.0 * IQR
        severe_upper = Q3 + 3.0 * IQR

        st.session_state.data = data[
            (data[col] >= severe_lower) & (data[col] <= severe_upper)
        ]
        st.success(f"Outliers severos removidos na coluna '{col}'.")

    elif method == "Substituir por Limites":
        # Substituir valores fora dos limites pelos próprios limites
        st.session_state.data[col] = data[col].clip(lower_bound, upper_bound)
        st.success(f"Valores substituídos pelos limites na coluna '{col}'.")

    elif method == "Substituir por Média":
        # Substituir valores fora dos limites pela média
        mean_value = data[col].mean()
        mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        st.session_state.data.loc[mask, col] = mean_value
        st.success(f"Valores substituídos pela média ({mean_value:.2f}) na coluna '{col}'.")

    elif method == "Substituir por Mediana":
        # Substituir valores fora dos limites pela mediana
        median_value = data[col].median()
        mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        st.session_state.data.loc[mask, col] = median_value
        st.success(f"Valores substituídos pela mediana ({median_value:.2f}) na coluna '{col}'.")

##########################################################
# FUNÇÃO DE GUARDAR O DATASET DEPOIS DO PRÉ-PROCESSAMENTO

def save_modified_dataset_in_memory():
    # Salvar o dataset tratado diretamente no session_state
    st.session_state.data_tratada = st.session_state.data.copy()  # Copiar o dataset tratado
    st.success("Dataset tratado foi salvo na memória para uso posterior.")

# Função de download 
def download_button(df, filename="dataset_tratado.csv"):
    """Função para permitir o download do dataset tratado em formato CSV"""
    csv = df.to_csv(index=False)
    buf = io.BytesIO()
    buf.write(csv.encode())
    buf.seek(0)
    
    st.download_button(
        label="Baixar Dataset Tratado",
        data=buf,
        file_name=filename,
        mime="text/csv"
    )


##########################################################
# FUNÇÃO DE RESUMO APÓS PRÉ-PROCESSAMENTO
class CustomPDF(FPDF):
    def header(self):
        # Baixar a imagem do logo e salvar localmente
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        response = requests.get(logo_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                tmpfile.write(response.content)
                tmpfile_path = tmpfile.name
                # Adicionar a imagem no cabeçalho
                self.image(tmpfile_path, 10, 8, 20) 
        else:
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, "Logo não disponível", align='C')
        
        # Definir fonte para o cabeçalho
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', align='C', ln=True)
        self.ln(15)  # Espaço após o cabeçalho

    def footer(self):
        # Ir para 1.5 cm da parte inferior
        self.set_y(-15)
        # Definir fonte para o rodapé
        self.set_font('Helvetica', 'I', 10)
        # Data atual
        current_date = datetime.now().strftime('%d/%m/%Y')
        # Adicionar rodapé com a data e número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', align='C')

# Função para gerar o PDF com a imagem da tabela
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Função para gerar o PDF com a tabela simplificada, correlação e boxplot
def generate_pdf_resumo(dataset, summary_df, missing_data, outlier_summary):
    def clean_text(text):
        if not isinstance(text, str):
            return text
        return text.encode('latin-1', errors='ignore').decode('latin-1')

    # Inicialização do PDF
    pdf = CustomPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)  

    # Título do Relatório
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Relatório Resumo dos Dados"), ln=True, align="C")
    pdf.ln(5)

    # Estatísticas Descritivas Simplificadas
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Estatísticas Descritivas"), ln=True)
    pdf.set_font("Helvetica", size=8)  

    # Criar um DataFrame simplificado com as colunas solicitadas: Nome da Coluna, Tipo de Dados, Count, e Média
    summary_simplified = pd.DataFrame({
        'Coluna': dataset.columns,
        'Tipo de Dados': dataset.dtypes,
        'Count': dataset.count(),
        'Top': dataset.mode().iloc[0],  # Valor mais frequente (top)
    })

    # Inicializar as colunas 'std', 'min' e 'max' como valores nulos
    summary_simplified['std'] = None
    summary_simplified['min'] = None
    summary_simplified['max'] = None
    summary_simplified['Média'] = None  # Para garantir que a média seja inicializada

    # Calcular as estatísticas apenas para as colunas numéricas
    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'Média'] = dataset[numeric_columns].mean()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'std'] = dataset[numeric_columns].std()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'min'] = dataset[numeric_columns].min()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'max'] = dataset[numeric_columns].max()

    # Formatar as colunas numéricas para 4 casas decimais
    for col in ['Média', 'std', 'min', 'max']:
        summary_simplified[col] = summary_simplified[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Substituir 'nan' por vazio
    summary_simplified = summary_simplified.fillna('')

    # Gerar a tabela diretamente no PDF
    pdf.set_fill_color(200, 220, 255)  # Cor de fundo do cabeçalho
    col_widths = [pdf.get_string_width(col) for col in summary_simplified.columns]  # Largura das colunas
    max_width = 180  # Largura máxima disponível (ajustável para caber na largura do PDF)

    # Ajustar largura das colunas proporcionalmente
    total_width = sum(col_widths)
    scale_factor = max_width / total_width
    col_widths = [width * scale_factor for width in col_widths]

    # Cabeçalho
    for i, col in enumerate(summary_simplified.columns):
        pdf.cell(col_widths[i], 10, clean_text(col), 1, 0, 'C', True)
    pdf.ln()

    # Linhas de dados
    for i, row in summary_simplified.iterrows():
        for j, cell in enumerate(row):
            pdf.cell(col_widths[j], 8, clean_text(str(cell)), 1, 0, 'C')
        pdf.ln()

    pdf.ln(10)  # Espaço após a tabela de estatísticas

    # Resumo de Valores Ausentes
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Resumo de Valores Ausentes"), ln=True)
    pdf.set_font("Helvetica", size=8)

    if not missing_data.empty:  # Verifica se os dados estão vazios
        # Tabela de Valores Ausentes
        missing_data_list = [(col, str(count)) for col, count in missing_data.items()]
        pdf.set_fill_color(200, 220, 255)  # Cor de fundo do cabeçalho
        pdf.cell(50, 10, clean_text("Variável"), 1, 0, 'C', True)
        pdf.cell(50, 10, clean_text("Total de Ausentes"), 1, 1, 'C', True)
        for col, count in missing_data_list:
            pdf.cell(50, 10, clean_text(col), 1)
            pdf.cell(50, 10, clean_text(count), 1, 1)
        pdf.ln(10)
    else:
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 10, txt=clean_text("Não há valores ausentes."), ln=True)
        pdf.ln(5)

    # Resumo de Outliers
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Resumo de Outliers"), ln=True)
    pdf.set_font("Helvetica", size=8)

    if outlier_summary:
        # Tabela de Outliers
        outlier_list = [(entry["Variável"], str(entry["Total de Outliers"])) for entry in outlier_summary]
        pdf.set_fill_color(200, 220, 255)  # Cor de fundo do cabeçalho
        pdf.cell(50, 10, clean_text("Variável"), 1, 0, 'C', True)
        pdf.cell(50, 10, clean_text("Total de Outliers"), 1, 1, 'C', True)
        for variable, total_outliers in outlier_list:
            pdf.cell(50, 10, clean_text(variable), 1)
            pdf.cell(50, 10, clean_text(total_outliers), 1, 1)
        pdf.ln(10)
    else:
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 10, txt=clean_text("Não há outliers."), ln=True)
        pdf.ln(75)

    # **Matriz de Correlação (Heatmap)**
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Matriz de Correlação das Variáveis"), ln=True)
    pdf.set_font("Helvetica", size=8)

    # Selecionar apenas as colunas numéricas para correlação
    numeric_data = dataset.select_dtypes(include=['float64', 'int64'])

    # Calcular a correlação
    correlation_matrix = numeric_data.corr()

    # Gerar o heatmap da correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", cbar=True, square=True)
    plt.title('Matriz de Correlação das Variáveis', fontsize=14, fontweight='bold')

    # Salvar o heatmap como imagem temporária
    temp_filename = "correlation_heatmap.png"
    plt.savefig(temp_filename)
    plt.close()

    # Adicionar o heatmap ao PDF
    pdf.image(temp_filename, x=10, w=180)
    pdf.ln(95)  # Ajustar o espaço após o gráfico

    # **Boxplot combinado de todas as variáveis numéricas**
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Boxplot das Variáveis Numéricas"), ln=True)
    pdf.set_font("Helvetica", size=8)

    # Gerar boxplot para todas as variáveis numéricas no mesmo gráfico
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_data)
    plt.title('Boxplot das Variáveis Numéricas')

    # Salvar o boxplot combinado como imagem temporária
    temp_filename_boxplot = "boxplot_combined.png"
    plt.savefig(temp_filename_boxplot)
    plt.close()

    # Adicionar o boxplot ao PDF
    pdf.image(temp_filename_boxplot, x=10, w=180)
    pdf.ln(75)  # Ajustar o espaço após o gráfico

    # **Salvar o PDF no buffer**
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1', errors='ignore')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# Função para salvar a tabela como imagem, com ajustes de formatação
def save_table_as_image(df, filename="table_image.png"):
    # Substituir `nan` por valores vazios
    df = df.fillna('')
    
    # Formatar valores numéricos para 4 casas decimais
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Gerar a imagem da tabela
    fig, ax = plt.subplots(figsize=(8, 4))  # Tamanho ajustado
    ax.axis('tight')
    ax.axis('off')

    # Criando a tabela com o estilo adequado
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colColours=['#D9EAF7'] * len(df.columns))  # Cor do cabeçalho da tabela

    # Ajustando o layout da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Tamanho da fonte
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Salvando a tabela como imagem
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()

# Resumo do Pré-processamento de dados:
def data_summary():
    st.subheader("Resumo dos Dados")

    # Usa diretamente st.session_state.data
    if 'data' in st.session_state and st.session_state.data is not None:
        dataset = st.session_state.data
        st.success("Usando o dataset tratado!")
    else:
        st.error("Nenhum dataset está disponível. Por favor, execute o tratamento de dados antes.")
        return

    # Verifica se há variáveis selecionadas
    selected_columns = st.session_state.get('selected_columns', [])
    if not selected_columns:
        selected_columns = dataset.columns.tolist()

    # Selecionar variáveis para exibição
    selected_columns_to_display = st.multiselect(
        "Selecione as variáveis para visualizar as estatísticas",
        options=selected_columns,
        default=selected_columns
    )

    # Informações gerais
    st.write("Número de linhas e colunas:", dataset[selected_columns_to_display].shape)

    # Filtra apenas as colunas numéricas (ignorando as categóricas)
    numeric_columns = dataset[selected_columns_to_display].select_dtypes(include=['number']).columns

    # Estatísticas Descritivas (calculando manualmente para cada tipo)
    data_types = dataset[selected_columns_to_display].dtypes
    summary_data = {
        'Count': dataset[selected_columns_to_display].count(),
        'Mean': dataset[numeric_columns].mean(),
        'Std': dataset[numeric_columns].std(),
        'Min': dataset[numeric_columns].min(),
        '25%': dataset[numeric_columns].quantile(0.25),
        '50%': dataset[numeric_columns].median(),
        '75%': dataset[numeric_columns].quantile(0.75),
        'Max': dataset[numeric_columns].max(),
    }

    # Transformar em DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df['Tipo de Dados'] = data_types

    # Arredondar os valores para 4 casas decimais
    summary_df = summary_df.round(4)

    # Preencher valores ausentes nas colunas numéricas com 0
    summary_df = summary_df.fillna(0)

    st.write("Estatísticas Descritivas e Tipos de Dados")
    st.dataframe(fix_dataframe_types(summary_df))
    
    # **Valores Ausentes**
    st.subheader("Resumo de Valores Ausentes")
    missing_data = dataset[selected_columns_to_display].isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.write("Valores ausentes encontrados:")
        st.dataframe(fix_dataframe_types(missing_data.rename("Total de Valores Ausentes")))
    else:
        st.write("Não há valores ausentes nas variáveis selecionadas.")

    # **Resumo de Outliers**
    st.subheader("Resumo de Outliers")
    numeric_data = dataset[selected_columns_to_display].select_dtypes(include=['number'])
    
    # Obter a lista de colunas já tratadas (se existir)
    treated_columns = st.session_state.get('treated_columns', [])
        
    if not numeric_data.empty:
        outlier_summary = []
        for column in numeric_data.columns:
            # Se a coluna já foi tratada, pula a análise
            if column in treated_columns:
                continue
                
            # Análise normal para colunas não tratadas
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            outliers = numeric_data[(numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)]
            if len(outliers) > 0:  # Adiciona apenas se houver outliers
                outlier_summary.append({
                    "Variável": column,
                    "Total de Outliers": len(outliers)
                })
    
        # Verifica se há outliers detectados
        if outlier_summary:
            st.dataframe(fix_dataframe_types(pd.DataFrame(outlier_summary)))
        else:
            st.write("Não há outliers nas variáveis selecionadas.")  # Mensagem quando não há outliers
    else:
        st.write("Nenhuma variável numérica para análise de outliers.")
    # **Boxplot** - Gráfico
    st.subheader("Boxplot das Variáveis Numéricas")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_data)
    plt.title('Boxplot das Variáveis Numéricas')
    st.pyplot(plt)

    # **Matriz de Correlação (Heatmap)**
    st.subheader("Matriz de Correlação das Variáveis")
    # Calcular a correlação
    correlation_matrix = numeric_data.corr()

    # Gerar o heatmap da correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", cbar=True, square=True)
    plt.title('Matriz de Correlação das Variáveis', fontsize=14, fontweight='bold', fontname='Helvetica')
    st.pyplot(plt)
    
    # **Função de Download do PDF**
    pdf_buffer = generate_pdf_resumo(dataset, summary_df, missing_data, outlier_summary)
    st.download_button(
        label="Baixar PDF com o Resumo",
        data=pdf_buffer,
        file_name="resumo_dos_dados.pdf",
        mime="application/pdf"
    )

    # Função de Download
    dataset_to_download = dataset[selected_columns_to_display]
    download_button(dataset_to_download)

    # Navegação
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Voltar"):
            st.session_state.step = 'outlier_detection'
            st.rerun()

    with col2:
        if st.button("Próxima etapa"):
            st.session_state.step = 'model_selection'
            st.rerun()

##########################################################
# FUNÇÃO DE MODELOS
def plot_metrics(metrics_df):
    try:
        # Inicializa a chave 'metrics' se não estiver no session_state
        if 'metrics' not in st.session_state:
            st.session_state['metrics'] = {}

        # Verificar se o DataFrame está vazio
        if metrics_df.empty:
            st.warning("Nenhum dado para plotar.")
            return

        # Armazenar métricas no session_state
        for _, row in metrics_df.iterrows():
            model_name = row.name  # Assumindo que o índice contém o nome do modelo
            st.session_state['metrics'][model_name] = row.to_dict()

        # Definir o índice do DataFramex
        metrics_df.set_index('Modelo', inplace=True)

        # Verificar se as colunas de classificação estão presentes
        classification_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        regression_columns = ['MSE', 'MAE', 'R²']

        if all(col in metrics_df.columns for col in classification_columns):
            # Plotar métricas de classificação
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df[classification_columns].plot(kind='bar', ax=ax)
            plt.title('Métricas de Desempenho dos Modelos (Classificação)', fontsize=16)
            plt.ylabel('Valor', fontsize=14)
            plt.xlabel('Modelos', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylim(0, 1)
            plt.legend(loc='lower right', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        elif all(col in metrics_df.columns for col in regression_columns):
            # Plotar métricas de regressão
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df[regression_columns].plot(kind='bar', ax=ax)
            plt.title('Métricas de Desempenho dos Modelos (Regressão)', fontsize=16)
            plt.ylabel('Valor', fontsize=14)
            plt.xlabel('Modelos', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        else:
            st.error("O DataFrame não contém métricas válidas para classificação ou regressão.")
            return

        # Mostrar o gráfico no Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro ao plotar as métricas: {str(e)}")
    
    finally:
        plt.clf()  # Limpar a figura para evitar sobreposições



# Adicionar os modelos de regressão na função get_default_param_grid
def get_default_param_grid(model_name):
    if model_name == "Support Vector Classification (SVC)":
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']  # Apenas para kernel 'rbf'
        }
    elif model_name == "K-Nearest Neighbors (KNN)":
        return {
            'n_neighbors': list(range(1, 21)),  # Testa todos os valores de 1 a 20
            'weights': ['uniform', 'distance']
        }
    elif model_name == "Random Forest":
        # Geração dinâmica de max_depth como range
        max_depth_range = [None] + list(range(5, 21, 5))  # [None, 5, 10, 15, 20]
        return {
            'max_depth': max_depth_range,
            'n_estimators': [10, 50, 100]
        }
    elif model_name == "Regressão por Vetores de Suporte (SVR)":
        return {
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.5],
            'kernel': ['linear', 'rbf']
        }
    elif model_name in ["Regressão Linear Simples (RLS)"]:
        return {}  # Regressão Linear geralmente não tem hiperparâmetros ajustáveis
    else:
        return {}


def configure_manual_params(model_key, param_grid, manual_params):
    """
    Configura manualmente os parâmetros para o modelo selecionado com intervalos personalizados exibidos.
    """
    st.write(f"Configurações manuais para o modelo: {model_key}")

    # **Limpar parâmetros inválidos no estado global ANTES de criar os widgets**
    if 'manual_params' in st.session_state and 'gamma' in st.session_state['manual_params']:
        del st.session_state['manual_params']['gamma']  # Remove 'gamma' do estado global

    # Intervalos específicos para parâmetros
    param_ranges = {
        'C': {'min': 0.1, 'max': 100.0, 'step': 0.1, 'default': 1.0},
        'epsilon': {'min': 0.01, 'max': 1.0, 'step': 0.01, 'default': 0.1},
        'gamma': {'min': 0.01, 'max': 1.0, 'step': 0.01, 'default': 0.1},
        'degree': {'min': 1, 'max': 5, 'step': 1, 'default': 3},
    }

    # Criar widgets para parâmetros
    for param in param_grid:
        # Parâmetros categóricos
        if isinstance(param_grid[param][0], str):
            manual_params[param] = st.selectbox(
                f"{param} (Opções: {', '.join(param_grid[param])}):",
                options=param_grid[param],
                index=0,
                key=f"{model_key}_{param}"
            )
        # Parâmetros numéricos
        elif isinstance(param_grid[param][0], (int, float)):
            param_type = float if any(isinstance(x, float) for x in param_grid[param]) else int

            # Verificar se existe intervalo personalizado
            if param in param_ranges:
                config = param_ranges[param]

                # Mostrar intervalo aceito como dica para o usuário
                st.write(f"**{param}** (Intervalo: {config['min']} a {config['max']})")

                # Configuração interativa
                if param == 'max_depth':  # Verifica se o parâmetro é 'max_depth'
                    manual_params[param] = st.selectbox(
                        f"{param}:",
                        options=[None] + list(range(1, 21)),  # Inclusão de None
                        index=0 if config['default'] is None else list(range(1, 21)).index(config['default']),
                        key=f"{model_key}_{param}"
                    )
                else:
                    # Para outros parâmetros numéricos
                    manual_params[param] = st.number_input(
                        f"{param}:",
                        min_value=config['min'],
                        max_value=config['max'],
                        value=config['default'],
                        step=config['step'],
                        key=f"{model_key}_{param}"
                    )

    # **Configuração dinâmica para 'gamma' com base no kernel**
    if 'kernel' in manual_params and manual_params['kernel'] == 'rbf':
        config = param_ranges['gamma']
        st.write(f"**gamma** (Intervalo: {config['min']} a {config['max']})")
        manual_params['gamma'] = st.number_input(
            "gamma:",
            min_value=config['min'],
            max_value=config['max'],
            value=config['default'],
            step=config['step'],
            key=f"{model_key}_gamma"
        )
    else:
        # **Remover 'gamma' do manual_params e do estado global se o kernel não for 'rbf'**
        manual_params.pop('gamma', None)
        if 'manual_params' in st.session_state and 'gamma' in st.session_state['manual_params']:
            del st.session_state['manual_params']['gamma']  # Remove também do estado global

    # Atualizar estado global com os parâmetros finais
    st.session_state['manual_params'] = manual_params
    st.session_state['best_params_str'] = json.dumps(manual_params, indent=2)

    # Diagnóstico: Exibir parâmetros salvos
    st.write("Parâmetros manuais salvos:", st.session_state['manual_params'])

    return manual_params




# Dicionário que mapeia modelos aos seus parâmetros válidos
VALID_PARAMS = {
    "Random Forest": ["n_estimators", "max_depth"],
    "Support Vector Classification (SVC)": ["C", "kernel", "gamma"],  # Agora inclui "gamma"
    "K-Nearest Neighbors (KNN)": ["n_neighbors", "weights"],
    "Regressão Linear Simples (RLS)": [],  # Sem hiperparâmetros ajustáveis
    "Regressão por Vetores de Suporte (SVR)": ["C", "epsilon", "kernel"],  # Parâmetros ajustáveis para SVR
}


# Função para configurar a validação cruzada com base na escolha do usuário
def get_cv_strategy(cv_choice, X_train, y_train):
    if cv_choice == "K-Fold":
        return KFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_choice == "Leave-One-Out":
        return LeaveOneOut()
    elif cv_choice == "Divisão em Treino e Teste":
        # Exemplo de divisão simples em treino e teste
        return train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    elif cv_choice == "Holdout":
        # Pode ser uma abordagem similar ao treino-teste com outro conjunto
        return train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=42)  # Default é K-Fold

def configure_svr(model_key, manual_params):
    st.write("Configuração de parâmetros para Support Vector Regression (SVR)")
    
    # Configurar parâmetros comuns
    c = st.number_input("Parâmetro C (Regularização)", min_value=0.1, max_value=100.0, step=0.1, value=1.0)
    epsilon = st.number_input("Parâmetro epsilon", min_value=0.0, max_value=1.0, step=0.1, value=0.1)
    kernel = st.selectbox("Kernel", options=["linear", "rbf", "poly", "sigmoid"], index=0)

    # Salvar os valores no dicionário de parâmetros
    manual_params['C'] = c
    manual_params['epsilon'] = epsilon
    manual_params['kernel'] = kernel

    # Configuração adicional para kernels específicos
    if kernel == "rbf":
        gamma = st.number_input("Parâmetro gamma", min_value=0.0, max_value=1.0, step=0.1, value=0.1)
        manual_params['gamma'] = gamma

    return manual_params

def configure_svc(model_key, manual_params):
    """Configura os parâmetros para o modelo SVC."""

    # Diagnóstico: Mostrar parâmetros antes da seleção manual
    st.write("Estado inicial dos parâmetros:", st.session_state.get('manual_params', {}))

    # Seleção do kernel
    kernel_value = st.selectbox(
        "Escolha o valor para 'kernel'",
        options=["linear", "rbf"],
        index=0,  # Define 'linear' como padrão
        key="kernel_selectbox"
    )

    # Configurar 'C' (sempre exibido)
    C_value = st.number_input(
        "Defina o valor para 'C'",
        min_value=0.01, step=0.01, value=1.0,
        key="C_input"
    )

    # Inicializa manual_params com 'C' e 'kernel'
    manual_params = {
        "C": C_value,
        "kernel": kernel_value
    }

    # **Exibir 'gamma' apenas se o kernel for 'rbf'**
    if kernel_value == "rbf":
        gamma_value = st.selectbox(
            "Escolha o valor para 'gamma'",
            options=["scale", "auto"],
            index=0,
            key="gamma_selectbox"
        )
        manual_params["gamma"] = gamma_value  # Adiciona gamma se necessário
    else:
        # **Remover 'gamma' se o kernel for 'linear'**
        # Remover do manual_params local
        manual_params.pop('gamma', None)
        # Remover do estado global
        if 'manual_params' in st.session_state and 'gamma' in st.session_state['manual_params']:
            del st.session_state['manual_params']['gamma']  # Remove globalmente
        if 'best_params_str' in st.session_state:  # Remove dos parâmetros salvos
            st.session_state['best_params_str'] = json.dumps(manual_params, indent=2)

    # Diagnóstico: Mostrar parâmetros após a seleção manual
    st.write("Parâmetros atualizados:", manual_params)

    # Salvar no estado global apenas parâmetros válidos
    st.session_state['manual_params'] = manual_params
    st.session_state['best_params_str'] = json.dumps(manual_params, indent=2)

    # Exibir os parâmetros salvos para depuração
    st.write("Parâmetros manuais salvos:", st.session_state['manual_params'])

    return manual_params

import pickle
import os

def save_best_params(params):
    """Salva os melhores parâmetros encontrados em um arquivo."""
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(params, f)

def load_best_params():
    """Carrega os melhores parâmetros salvos, se existirem."""
    if os.path.exists('best_params.pkl'):
        with open('best_params.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def train_model_with_gridsearch(model, param_grid, X_train, y_train, use_grid_search, manual_params=None, cv_choice="K-Fold"):
    try:
        # Inicializar parâmetros manuais como vazio, se não fornecido
        if manual_params is None:
            manual_params = {}
        # Verifica se max_depth está no param_grid antes de iniciar o GridSearch
        print("Parâmetros antes do GridSearchCV:", param_grid)
        # Obter o nome do modelo
        model_name = None
        if isinstance(model, SVC):
            model_name = "Support Vector Classification (SVC)"
        elif isinstance(model, KNeighborsClassifier):
            model_name = "K-Nearest Neighbors (KNN)"
        elif isinstance(model, RandomForestClassifier):
            model_name = "Random Forest"
        elif isinstance(model, SVR):
            model_name = "Regressão por Vetores de Suporte (SVR)"
        else:
            st.error(f"Modelo não reconhecido: {model}")
            return None, None

        # Logs para diagnóstico - Parâmetros no estado global antes do treino
        st.write("Parâmetros no estado global antes do treino:")
        st.write("best_params:", st.session_state.get('best_params', {}))
        st.write("manual_params:", st.session_state.get('manual_params', {}))

        # Carregar parâmetros salvos do estado global
        saved_params = st.session_state.get('best_params', None)

        # **Aplicar parâmetros salvos, se existirem**
        if saved_params and not use_grid_search:  # Se não usar GridSearch, aplica os salvos
            st.info(f"Aplicando parâmetros salvos ao modelo: {saved_params}")
            model.set_params(**saved_params)

        # Remover 'gamma' se o kernel for 'linear'
        if manual_params.get("kernel") == "linear" and "gamma" in manual_params:
            del manual_params["gamma"]  # Remove localmente
            if 'gamma' in st.session_state.get('manual_params', {}):
                del st.session_state['manual_params']['gamma']  # Remove globalmente

        # **Se usar GridSearch**
        if use_grid_search:
            # Atualizar grid com parâmetros manuais fornecidos
            if manual_params:
                for param, value in manual_params.items():
                    if not isinstance(value, list):
                        manual_params[param] = [value]  # Converter para lista

                # Atualiza o param_grid
                param_grid.update(manual_params)

            # Configurar validação cruzada
            cv_strategy = get_cv_strategy(cv_choice, X_train, y_train)
            scoring = 'r2' if model_name == "Regressão por Vetores de Suporte (SVR)" else 'accuracy'

            # Treinar com GridSearch
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Melhor modelo e parâmetros
            best_params = grid_search.best_params_
            st.session_state['best_params'] = best_params  # Salva no estado global
            st.success(f"Melhores parâmetros encontrados: {best_params}")
            st.write("Parâmetros salvos no estado global:", st.session_state['best_params'])

            # Retorna o melhor modelo e parâmetros
            return grid_search.best_estimator_, best_params

        else:
            # Se não usar GridSearch, aplicar manualmente os parâmetros
            if manual_params:
                valid_params = model.get_params().keys()
                manual_params = {k: v for k, v in manual_params.items() if k in valid_params}
                model.set_params(**manual_params)

            # Treinar diretamente
            model.fit(X_train, y_train)

            # Salvar parâmetros manuais no estado global
            st.session_state['manual_params'] = manual_params
            st.success(f"Parâmetros manuais salvos: {manual_params}")

            # Retorna o modelo e parâmetros manuais
            return model, manual_params

    except Exception as e:
        st.error(f"Ocorreu um erro ao treinar o modelo: {str(e)}")
        return None, None


# Função para calcular o Gap Statistic para o Clustering Hierárquico
def calculate_gap_statistic_hierarchical(X, n_clusters_range, n_ref=10):
    """
    Calcula o Gap Statistic para o AgglomerativeClustering.
    
    Parâmetros:
        X (ndarray): Dados de entrada (n_samples x n_features).
        n_clusters_range (tuple): Intervalo de números de clusters para avaliar.
        n_ref (int): Número de amostras de referência aleatórias a serem geradas.
    
    Retorna:
        gap_scores (list): Gap statistics para cada número de clusters.
    """
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Armazenar os Gap Statistic Scores
    gap_scores = []
    
    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        # Ajustar o modelo AgglomerativeClustering aos dados reais
        model = AgglomerativeClustering(n_clusters=n_clusters)
        model.fit(X_scaled)
        labels = model.labels_
        
        # Calcular a soma das distâncias dos pontos aos seus respectivos clusters
        intra_cluster_dist = sum([np.sum(np.linalg.norm(X_scaled[labels == i] - X_scaled[labels == i].mean(axis=0), axis=1)) for i in range(n_clusters)])
        
        # Gerar amostras de referência aleatórias e calcular as distâncias dentro dos clusters aleatórios
        ref_inertias = []
        for _ in range(n_ref):
            random_data = np.random.random_sample(size=X_scaled.shape)
            random_model = AgglomerativeClustering(n_clusters=n_clusters)
            random_model.fit(random_data)
            ref_labels = random_model.labels_
            ref_inertia = sum([np.sum(np.linalg.norm(random_data[ref_labels == i] - random_data[ref_labels == i].mean(axis=0), axis=1)) for i in range(n_clusters)])
            ref_inertias.append(ref_inertia)
        
        # Calcular a média e o desvio padrão das inércias nos dados aleatórios
        ref_inertia_mean = np.mean(ref_inertias)
        ref_inertia_std = np.std(ref_inertias)
        
        # Gap Statistic: diferença entre a inércia real e a média das inércias aleatórias
        gap = np.log(ref_inertia_mean) - np.log(intra_cluster_dist)
        gap_scores.append(gap)
    
    return gap_scores



# Função para a seleção e treino de modelos
def model_selection():
    st.subheader("Seleção e treino de Modelos")

    # Verificar se os dados estão disponíveis
    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("Dados não encontrados. Por favor, carregue os dados primeiro.")
        return

    # Usa diretamente st.session_state.data
    data = st.session_state.data
    columns = data.columns.tolist()
    
    # Inicializar variáveis de estado se não estiverem presentes
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'target_column_confirmed' not in st.session_state:
        st.session_state.target_column_confirmed = False
    if 'validation_method' not in st.session_state:
        st.session_state.validation_method = None
    if 'validation_confirmed' not in st.session_state:
        st.session_state.validation_confirmed = False
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'model_type_confirmed' not in st.session_state:
        st.session_state.model_type_confirmed = False
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_selection_done' not in st.session_state:
        st.session_state.feature_selection_done = False

    # Configurações
    st.write("### Configurações")


    # 1. Escolha do Tipo de Modelo
    if not st.session_state.model_type_confirmed:
        st.write("Escolha o Tipo de Modelo")
        model_types = ["Classificação", "Regressão", "Clustering"]
        st.session_state.model_type = st.selectbox("Selecione o tipo de modelo", model_types)

        if st.button("Confirmar Tipo de Modelo"):
            st.session_state.model_type_confirmed = True
            st.success("Tipo de modelo confirmado!")

    # 2. Escolha do Modelo Específico
    if st.session_state.model_type_confirmed and not st.session_state.selected_model_name:
        st.write("Selecione o(s) Modelo(s)")

        # Modelos disponíveis com base no tipo selecionado
        if st.session_state.model_type == "Classificação":
            models = {
                "Support Vector Classification (SVC)": SVC(),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier()
            }
        elif st.session_state.model_type == "Regressão":
            models = {
                "Regressão Linear Simples (RLS)": LinearRegression(),
                "Regressão por Vetores de Suporte (SVR)": SVR(),
            }
        elif st.session_state.model_type == "Clustering":
            models = {
                "KMeans": KMeans(),
                "Clustering Hierárquico": AgglomerativeClustering(),
            }

        # Armazena os modelos no session_state para uso posterior
        st.session_state.models = models
        

        # Condicional para exibir ou não a opção "Treinar todos os modelos"
        if st.session_state.model_type != "Clustering":
            model_options = list(models.keys()) 
        else:
            model_options = list(models.keys())  # Apenas os modelos de clustering

        default_model_name = st.session_state["model_name"]
        if default_model_name not in model_options:
            default_model_name = model_options[0]  # Corrigir para um valor válido

        # Configurar o selectbox
        model_name = st.selectbox(
            "Selecione o modelo", 
            options=model_options, 
            key="model_name_selectbox", 
            index=model_options.index(default_model_name)
        )

        # Atualizar o estado do modelo selecionado
        st.session_state["model_name"] = model_name
        st.session_state.model_name = model_name

        # Botão para confirmar o modelo
        if st.button("Confirmar Modelo"):
            if model_name:  # Verifica se um modelo foi selecionado
                st.session_state.selected_model_name = model_name
                st.success(f"Modelo selecionado: {st.session_state.selected_model_name}")
            else:
                st.warning("Selecione um modelo antes de continuar.")



    # Função para a configuração de Clustering
    import pandas as pd

    # Inicializar a variável `best_n_clusters_retrain` com um valor padrão
    best_n_clusters_retrain = None

    # Função para a configuração de Clustering
    if st.session_state.model_type == "Clustering" and st.session_state.selected_model_name:
        st.write("### Configuração para Clustering")

        # Escolher o intervalo de clusters
        num_clusters_range = st.slider("Intervalo de clusters para explorar (para análise)", 2, 20, (2, 10))

        # Dados categóricos codificados
        X = pd.get_dummies(st.session_state.data)

        # Análise de clusters
        st.write("### Análise para Determinação do Número de Clusters")
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        # Condicional para KMeans
        if st.session_state.selected_model_name == "KMeans":
            for n_clusters in range(num_clusters_range[0], num_clusters_range[1] + 1):
                temp_model = KMeans(n_clusters=n_clusters, random_state=42)
                temp_model.fit(X)
                labels = temp_model.labels_

                # Calcular as métricas e armazená-las
                silhouette_scores.append(silhouette_score(X, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

            # Criar DataFrame com os resultados
            metrics_df = pd.DataFrame({
                "Número de Clusters": range(num_clusters_range[0], num_clusters_range[1] + 1),
                "Silhouette Score": silhouette_scores,
                "Davies-Bouldin Index": davies_bouldin_scores,
                "Calinski-Harabasz Score": calinski_harabasz_scores,
            })

            # Exibir a tabela no Streamlit
            st.write("#### Tabela de Métricas por Número de Clusters")
            st.dataframe(fix_dataframe_types(metrics_df.style.format({
                "Silhouette Score": "{:.2f}",
                "Davies-Bouldin Index": "{:.2f}",
                "Calinski-Harabasz Score": "{:.2f}",
            })))

            # Exibir gráficos para as métricas
            st.write("#### Gráficos das Métricas por Número de Clusters")
            col1, col2, col3 = st.columns(3)

            with col1:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Silhouette Score"], marker='o')
                plt.title("Silhouette Score por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Silhouette Score")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Davies-Bouldin Index"], marker='o')
                plt.title("Davies-Bouldin Index por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Davies-Bouldin Index")
                st.pyplot(plt.gcf())
                plt.clf()

            with col3:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Calinski-Harabasz Score"], marker='o')
                plt.title("Calinski-Harabasz Score por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Calinski-Harabasz Score")
                st.pyplot(plt.gcf())
                plt.clf()

            # Escolher o melhor número de clusters com base no Silhouette Score (ou qualquer outra métrica escolhida)
            best_n_clusters = metrics_df.loc[metrics_df["Silhouette Score"].idxmax(), "Número de Clusters"]
            st.write(f"**Melhor Número de Clusters** (com base no Silhouette Score): {best_n_clusters}")

        # Condicional para Clustering Hierárquico
        elif st.session_state.selected_model_name == "Clustering Hierárquico":
            for n_clusters in range(num_clusters_range[0], num_clusters_range[1] + 1):
                temp_model = AgglomerativeClustering(n_clusters=n_clusters)
                temp_model.fit(X)
                labels = temp_model.labels_

                # Calcular as métricas e armazená-las
                silhouette_scores.append(silhouette_score(X, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

            # Criar DataFrame com os resultados
            metrics_df = pd.DataFrame({
                "Número de Clusters": range(num_clusters_range[0], num_clusters_range[1] + 1),
                "Silhouette Score": silhouette_scores,
                "Davies-Bouldin Index": davies_bouldin_scores,
                "Calinski-Harabasz Score": calinski_harabasz_scores,
            })

            # Exibir a tabela no Streamlit
            st.write("#### Tabela de Métricas por Número de Clusters")
            st.dataframe(fix_dataframe_types(metrics_df.style.format({
                "Silhouette Score": "{:.2f}",
                "Davies-Bouldin Index": "{:.2f}",
                "Calinski-Harabasz Score": "{:.2f}",
            })))

            # Exibir gráficos para as métricas
            st.write("#### Gráficos das Métricas por Número de Clusters")
            col1, col2, col3 = st.columns(3)

            with col1:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Silhouette Score"], marker='o')
                plt.title("Silhouette Score por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Silhouette Score")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Davies-Bouldin Index"], marker='o')
                plt.title("Davies-Bouldin Index por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Davies-Bouldin Index")
                st.pyplot(plt.gcf())
                plt.clf()

            with col3:
                plt.figure(figsize=(6, 4))
                plt.plot(metrics_df["Número de Clusters"], metrics_df["Calinski-Harabasz Score"], marker='o')
                plt.title("Calinski-Harabasz Score por Número de Clusters")
                plt.xlabel("Número de Clusters")
                plt.ylabel("Calinski-Harabasz Score")
                st.pyplot(plt.gcf())
                plt.clf()

            # Escolher o melhor número de clusters com base no Silhouette Score (ou outra métrica)
            best_n_clusters = metrics_df.loc[metrics_df["Silhouette Score"].idxmax(), "Número de Clusters"]
            st.write(f"**Melhor Número de Clusters** (com base no Silhouette Score): {best_n_clusters}")

            # Inicializar `best_n_clusters_retrain` com o valor automático, caso aplicável
            best_n_clusters_retrain = best_n_clusters

        # Escolher abordagem para número de clusters
        st.write("### Escolha a Abordagem para Determinar o Número de Clusters")
        method = st.radio("Selecione a abordagem:", ["Automático", "Manual"], key="initial_training_method")

        if method == "Automático":
            # Escolher o melhor número de clusters com base no Silhouette Score
            best_n_clusters = range(num_clusters_range[0], num_clusters_range[1] + 1)[np.argmax(silhouette_scores)]
            best_n_clusters_retrain = best_n_clusters  # Atualizar o valor para re-treino

        elif method == "Manual":
            best_n_clusters = st.slider("Escolha o número de clusters", num_clusters_range[0], num_clusters_range[1], value=3)
            best_n_clusters_retrain = best_n_clusters  # Atualizar o valor para re-treino

        # Garantir que `best_n_clusters_retrain` tenha um valor válido antes de usar
        if best_n_clusters_retrain is None:
            st.warning("Por favor, selecione uma abordagem para determinar o número de clusters.")
        else:
            # Treinar modelo inicial
            if st.button(f"Treinar Modelo Inicial"):
                model = st.session_state.models[st.session_state.selected_model_name]
                model.set_params(n_clusters=best_n_clusters_retrain)
                model.fit(X)
                st.session_state.clustering_labels = model.labels_
                st.session_state.initial_metrics = {
                    "Número de Clusters": best_n_clusters_retrain,
                    "Silhouette Score": silhouette_score(X, st.session_state.clustering_labels),
                    "Davies-Bouldin Index": davies_bouldin_score(X, st.session_state.clustering_labels),
                    "Calinski-Harabasz Score": calinski_harabasz_score(X, st.session_state.clustering_labels)
                }
                st.success(f"Modelo inicial treinado com sucesso usando {best_n_clusters_retrain} clusters!")
                st.session_state.training_completed = True  # Marcar como concluído

        # Exibir métricas e próxima ação apenas após o treino
        if st.session_state.get("training_completed", False):
            st.write("### Métricas do Treino Inicial")
            st.table(fix_dataframe_types(pd.DataFrame([st.session_state.initial_metrics])))

            # Escolher ação seguinte
            next_action = st.selectbox(
                "Selecione a próxima ação:",
                ["Re-Treinar o Modelo", "Finalizar"]
            )

            # Botão de confirmação da escolha
            if st.button("Confirmar Escolha"):
                if next_action == "Finalizar":
                    st.session_state.step = 'clustering_final_page'
                    st.rerun()
                elif next_action == "Re-Treinar o Modelo":
                    st.session_state.retrain_mode = True

        # Re-Treinar o Modelo (só aparece se a escolha foi confirmada)
        if st.session_state.get("retrain_mode", False):
            retrain_method = st.radio(
                "Escolha a Abordagem para Determinar o Número de Clusters no novo treino:",
                ["Automático", "Manual"]
            )

            if retrain_method == "Manual":
                st.session_state.num_clusters = st.slider(
                    "Selecione o número de clusters para o re-treino",
                    min_value=2,
                    max_value=20,
                    value=st.session_state.num_clusters if "num_clusters" in st.session_state else 3
                )
                best_n_clusters_retrain = st.session_state.num_clusters

            elif retrain_method == "Automático":
                # Determinar o melhor número de clusters com base no Silhouette Score
                best_n_clusters_retrain = range(num_clusters_range[0], num_clusters_range[1] + 1)[np.argmax(silhouette_scores)]

            # Botão para executar o re-treino
            if st.button("Treinar Novamente"):
                model = st.session_state.models[st.session_state.selected_model_name]
                model.set_params(n_clusters=best_n_clusters_retrain)
                model.fit(X)
                st.session_state.retrain_metrics = {
                    "Número de Clusters": best_n_clusters_retrain,
                    "Silhouette Score": silhouette_score(X, model.labels_),
                    "Davies-Bouldin Index": davies_bouldin_score(X, model.labels_),
                    "Calinski-Harabasz Score": calinski_harabasz_score(X, model.labels_)
                }
                st.session_state.retrain_completed = True
                st.success(f"Re-treino concluído com sucesso com {best_n_clusters_retrain} clusters!")

            # Exibir métricas do re-treino após a execução
            if st.session_state.get("retrain_completed", False):
                st.write("### Métricas do Re-Treino")
                st.table(fix_dataframe_types(pd.DataFrame([st.session_state.retrain_metrics])))

            # Finalizar após o re-treino
            if st.session_state.get("retrain_completed", False):
                st.write("## Concluir o Processo de Clustering.")
                if st.button("Seguir para o Relatório"):
                    st.session_state.step = 'clustering_final_page'
                    st.rerun()

                        
    # 3. Seleção da Coluna Alvo
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    # Inicializar variáveis de estado
    if 'bins_confirmed' not in st.session_state:
        st.session_state['bins_confirmed'] = False  # Confirmação dos bins
    if 'bins_value' not in st.session_state:
        st.session_state['bins_value'] = 3  # Valor padrão dos bins

    # Filtrar colunas disponíveis com base no tipo de modelo
    if st.session_state.model_type == "Classificação":
        valid_columns = [col for col in columns if data[col].dtype in ['object', 'int64'] or data[col].nunique() <= 10]
    else:
        valid_columns = [col for col in columns if data[col].dtype in ['float64', 'int64'] and data[col].nunique() > 10]

    # Seleção da Coluna Alvo
    if st.session_state.model_type != "Clustering" and st.session_state.selected_model_name and not st.session_state.target_column_confirmed:
        st.write("Escolha a Coluna Alvo")
        target_column = st.selectbox(
            "Selecione a coluna alvo",
            options=valid_columns,
            key='target_column_selectbox'
        )

        if st.button("Confirmar Coluna Alvo"):
            if target_column in columns:
                st.session_state.target_column = target_column
                st.session_state.target_column_confirmed = True
                st.session_state.validation_method = None
                st.session_state.validation_confirmed = False

                # Processar a coluna alvo
                y = data[st.session_state.target_column]

                # Verificar o tipo de modelo
                model_type = st.session_state.model_type

                # **Se o modelo for de Classificação**
                if model_type == "Classificação":
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    st.session_state['target_column_encoded'] = y_encoded
                    st.success("Coluna categórica detectada e codificada com LabelEncoder.")

                elif model_type == "Regressão":
                    if y.dtype in ['float64', 'int64']:
                        st.session_state['target_column_encoded'] = y
                        st.success("Coluna contínua detectada e pronta para regressão.")
                    else:
                        st.error("Modelos de regressão requerem uma coluna contínua como alvo.")
                        st.stop()



    # Exibir a Coluna Alvo Confirmada
    if st.session_state.model_type != "Clustering" and st.session_state.target_column_confirmed:
        st.write(f"Coluna Alvo Confirmada: {st.session_state.target_column}")
        st.write(f"Tipo: {st.session_state.get('target_column_type', 'Não definido')}")


        # 4. GridSearch
        # Modelos sem hiperparâmetros ajustáveis
        def limpar_parametros_invalidos():
            """Remove parâmetros inválidos do session_state."""
            if 'manual_params' in st.session_state:
                if 'gamma' in st.session_state['manual_params']:
                    del st.session_state['manual_params']['gamma']  # Remove 'gamma' se presente

        # Inicializa modelos sem hiperparâmetros ajustáveis
        NO_HYPERPARAM_MODELS = ["Regressão Linear Simples (RLS)"]

        # Verifica se o modelo foi selecionado
        if st.session_state.selected_model_name and not st.session_state.grid_search_confirmed:

            # Verificar se o modelo não possui hiperparâmetros ajustáveis
            if st.session_state.selected_model_name in NO_HYPERPARAM_MODELS:
                st.write(f"O modelo {st.session_state.selected_model_name} não possui hiperparâmetros ajustáveis.")
                st.session_state.use_grid_search = "Não"
                param_grid = {}  # Nenhum parâmetro para ajustar
                st.session_state.grid_search_confirmed = True
            else:
                # Perguntar ao usuário se quer usar GridSearch
                use_grid_search = st.radio(
                    "Usar GridSearch?", 
                    ["Sim", "Não"], 
                    key='grid_search_radio', 
                    index=0 if st.session_state.get('use_grid_search', "Sim") == "Sim" else 1
                )
                st.session_state.use_grid_search = use_grid_search

                # Inicializar param_grid como vazio
                param_grid = {}  # Evita erros de variável não definida

                if use_grid_search == "Sim":
                    # Perguntar como os parâmetros devem ser escolhidos
                    param_choice = st.radio(
                        "Escolher os parâmetros de GridSearch?",
                        ["Utilizar os melhores parâmetros", "Escolher manualmente os parâmetros de GridSearch"],
                        key='param_choice_radio',
                        index=0 if st.session_state.get('param_choice', "Utilizar os melhores parâmetros") == "Utilizar os melhores parâmetros" else 1
                    )
                    st.session_state.param_choice = param_choice

                    # Inicializar parâmetros manuais
                    if 'manual_params' not in st.session_state:
                        st.session_state.manual_params = {}

                    manual_params = st.session_state.manual_params

                    # Configuração manual dos parâmetros
                    if param_choice == "Escolher manualmente os parâmetros de GridSearch":
                        # Recuperar o modelo selecionado
                        model_key = st.session_state.selected_model_name
                    
                        # Inicializar os parâmetros padrão do modelo selecionado
                        param_grid = get_default_param_grid(model_key)
                    
                        # Se não houver parâmetros padrão, informar o usuário
                        if not param_grid:
                            st.warning(f"Parâmetros padrão não definidos para o modelo {model_key}.")
                            param_grid = {}
                    
                        # Exibir os parâmetros para o usuário ajustar manualmente
                        manual_params = {}
                        for param, values in param_grid.items():
                            # **Lógica Especial para o Kernel**
                            if param == "kernel":
                                # Selecionar o kernel
                                manual_params[param] = st.selectbox(
                                    f"Escolha o valor para '{param}':",
                                    values,  # Lista de valores permitidos
                                    index=0,  # Primeiro valor como padrão
                                    key=f"{model_key}_{param}"
                                )
                    
                            # **Mostrar 'gamma' apenas se o kernel for 'rbf'**
                            elif param == "gamma":
                                if "kernel" in manual_params and manual_params["kernel"] == "rbf":
                                    # Mostrar gamma apenas para 'rbf'
                                    manual_params[param] = st.selectbox(
                                        f"Escolha o valor para '{param}':",
                                        values,  # Lista de valores permitidos
                                        index=0,  # Primeiro valor como padrão
                                        key=f"{model_key}_{param}"
                                    )
                                else:
                                    # Remover 'gamma' do estado global e local
                                    manual_params.pop(param, None)
                                    if 'manual_params' in st.session_state and param in st.session_state['manual_params']:
                                        del st.session_state['manual_params'][param]
                    
                            # **Tratar parâmetros numéricos**
                            elif isinstance(values[0], (int, float)):
                                # Mostrar os valores disponíveis para o parâmetro
                                st.write(f"Parâmetro: **{param}** | Intervalo disponível: [{min(values)}, {max(values)}]")
                            
                                # Verificar o tipo de dado (float ou int) para parametrização
                                param_type = float if any(isinstance(v, float) for v in values) else int
                            
                                # Criar o número interativo
                                manual_params[param] = st.number_input(
                                    f"Escolha o valor para '{param}':",
                                    min_value=float(min(values)) if param_type == float else int(min(values)),
                                    max_value=float(max(values)) if param_type == float else int(max(values)),
                                    value=float(values[0]) if param_type == float else int(values[0]),
                                    step=0.1 if param_type == float else 1,  # Ajuste o step dinamicamente
                                    key=f"{model_key}_{param}"
                                )
                            
                            # **Tratar `max_depth` separadamente como um selectbox**
                            elif param == "max_depth":
                                st.write(f"Parâmetro: **{param}** | Valores disponíveis: {values}")
                                manual_params[param] = st.selectbox(
                                    f"Escolha o valor para '{param}':",
                                    values,
                                    index=0,  # Primeiro valor como padrão
                                    key=f"{model_key}_{param}"
                                )
                    
                            # **Tratar parâmetros categóricos (ex.: 'weights')**
                            elif isinstance(values[0], str):
                                # Mostrar os valores disponíveis para o parâmetro
                                st.write(f"Parâmetro: **{param}** | Valores disponíveis: {values}")
                            
                                # Criar o selectbox interativo
                                manual_params[param] = st.selectbox(
                                    f"Escolha o valor para '{param}':",
                                    values,  # Lista de valores permitidos
                                    index=0,  # Primeiro valor como padrão
                                    key=f"{model_key}_{param}"
                                )
                    
                        # Salvar os parâmetros manuais no estado global
                        st.session_state['manual_params'] = manual_params
                        st.write("Parâmetros manuais salvos:", manual_params)



                # Confirmar configurações do GridSearch
                if st.button("Confirmar GridSearch"):
                    st.session_state.grid_search_confirmed = True
                    st.success("Configuração do GridSearch confirmada!")

                    # Parâmetros padrão até o treino
                    if st.session_state.use_grid_search == "Sim" and st.session_state.param_choice == "Utilizar os melhores parâmetros":
                        st.session_state['manual_params'] = {}
                        st.session_state['best_params_str'] = "{}"
                        st.session_state['best_params'] = param_grid
                        st.session_state['best_params_selected'] = param_grid
                        

        # 5. Escolha do Método de Validação
        # O método de validação agora aparece somente após confirmação do GridSearch
        if st.session_state.grid_search_confirmed and st.session_state.selected_model_name and not st.session_state.validation_method:
            st.write("Escolha o Método de Validação")
            validation_methods = ["Divisão em Treino e Teste", "Holdout"]
            validation_method = st.radio(
                "Escolha o método de validação",
                validation_methods,
                key='validation_method_radio'
            )

            # Configurações específicas para cada método de validação
            if validation_method == "Divisão em Treino e Teste":
                test_size = st.slider(
                    "Proporção do conjunto de teste",
                    min_value=0.1, max_value=0.9, value=0.3, step=0.1
                )
                st.session_state.test_size = test_size

            elif validation_method == "Holdout":
                train_size = st.slider(
                    "Proporção do conjunto de treino",
                    min_value=0.1, max_value=0.9, value=0.7, step=0.1
                )
                st.session_state.train_size = train_size

            # Botão de confirmação para o método de validação
            if st.button("Confirmar Validação"):
                st.session_state.validation_method = validation_method  # Armazena o método de validação escolhido

                # Preparação de dados para validação
                X = data.drop(columns=[st.session_state.target_column])
                y = data[st.session_state.target_column]

                # Conversão de variáveis categóricas para numéricas
                X = pd.get_dummies(X)

                try:
                    # Tratamento de diferentes métodos de validação
                    if st.session_state.validation_method == "Divisão em Treino e Teste":
                        # Divisão simples em treino e teste
                        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                            X, y, test_size=st.session_state.test_size, random_state=42
                        )
                        st.success("Divisão dos dados realizada com sucesso!")

                    elif st.session_state.validation_method == "Holdout":
                        # Holdout: outra forma de divisão de treino e teste
                        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                            X, y, train_size=st.session_state.train_size, random_state=42
                        )
                        st.success("Divisão dos dados realizada com sucesso!")

                    # Confirma a validação
                    st.session_state.validation_confirmed = True

                except Exception as e:
                    st.error(f"Erro na divisão dos dados: {e}")

                # Exibir método de validação confirmado
                if st.session_state.validation_confirmed:
                    st.write(f"Método de Validação Confirmado: {st.session_state.validation_method}")

        # Exibir o botão para treinar o modelo **apenas após a validação ser confirmada**
        # 6. Treino do Modelo
        if st.session_state.validation_confirmed:
            if st.button("Treinar o Modelo"):
                st.session_state.validation_confirmed = False  # Resetando após o treino
                st.success("Treino iniciado com sucesso!")

                # Recuperar o modelo selecionado
                model_name = st.session_state.selected_model_name
                model = st.session_state.models.get(st.session_state.selected_model_name)

                # Verificar se o modelo foi encontrado
                if model is None:
                    st.error(f"Modelo {st.session_state.selected_model_name} não encontrado.")
                    return  # Interrompe o fluxo caso o modelo não seja encontrado

                # Inicializar 'treinos_realizados' se necessário
                if 'treinos_realizados' not in st.session_state:
                    st.session_state['treinos_realizados'] = []

                # Coletar as informações armazenadas no session_state
                target_column = st.session_state.target_column
                validation_method = st.session_state.validation_method
                use_grid_search = st.session_state.use_grid_search
                manual_params = st.session_state.manual_params
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                # **Remover parâmetros inválidos antes do treino**
                if 'manual_params' in st.session_state:
                    if manual_params.get('kernel') == 'linear' and 'gamma' in manual_params:
                        del manual_params['gamma']  # Remove o parâmetro local
                    if 'gamma' in st.session_state['manual_params']:
                        del st.session_state['manual_params']['gamma']  # Remove do estado global

                # **Adicionar tratamento de valores ausentes**
                from sklearn.impute import SimpleImputer

                imputer = SimpleImputer(strategy="mean")  # Ou "median" conforme necessário
                X_train = imputer.fit_transform(X_train)  # Tratamento no conjunto de treino
                X_test = imputer.transform(X_test)        # Tratamento no conjunto de teste

                # Exibir resumo das escolhas feitas antes do treino
                st.write("### Resumo das Escolhas Feitas:")
                st.write(f"**Modelo Selecionado**: {model_name}")
                st.write(f"**Coluna Alvo**: {target_column}")
                st.write(f"**Método de Validação**: {validation_method}")
                st.write(f"GridSearch Ativado? {use_grid_search}")  # Debug para verificar a escolha do usuário

                # Treino de um único modelo
                param_grid = get_default_param_grid(model_name) if use_grid_search == "Sim" else {}
                resultado = train_and_evaluate(
                    model, param_grid, X_train, y_train, X_test, y_test, use_grid_search, manual_params
                )

                # **Salvar apenas os parâmetros válidos no estado global após o treino**
                if 'Best Parameters' in resultado:
                    st.session_state['best_params'] = resultado['Best Parameters']  # Para treino inicial
                    st.session_state['best_params_selected'] = resultado['Best Parameters']  # Para seleção de features
                    st.session_state['best_params_str'] = json.dumps(st.session_state['best_params'], indent=2)
                    st.write("Parâmetros salvos no estado global:", st.session_state['best_params'])
                else:
                    st.warning("Nenhum parâmetro encontrado para salvar.")

                # Após o primeiro treino
                # Após o primeiro treino
                if resultado:
                    # Armazena os resultados iniciais para comparação futura
                    st.session_state['resultado_sem_selecao'] = resultado  # Salva os resultados sem seleção
                    st.session_state['treinos_realizados'].append(resultado)
                    
                    # Criar o DataFrame com as métricas
                    df_resultado = pd.DataFrame([resultado])
                
                    # Corrigir os tipos antes de formatar
                    df_corrigido = fix_dataframe_types(df_resultado)
                    
                    # Aplicar formatação depois de corrigir os tipos
                    st.write("Métricas do modelo treinado:")
                    formatted_display = df_corrigido.style.format(
                        {col: "{:.4f}" for col in df_corrigido.select_dtypes(include=['float', 'float64']).columns}
                    )
                    st.dataframe(formatted_display)
                
                    # Gráfico das métricas
                    plot_metrics(df_corrigido)
                
                    # Marcar o treino como concluído
                    st.session_state['treino_concluido'] = True
                else:
                    st.error("O treino do modelo falhou.")

        # Avançar para Seleção de Features SOMENTE após o gráfico de métricas ser mostrado
        if st.session_state.get('treino_concluido', False):
            st.write("### Avançar para Seleção de Features")

            # Garantir que há treinos realizados
            if 'treinos_realizados' in st.session_state and st.session_state['treinos_realizados']:
                # Depuração: Exibir treinos realizados
                #st.write("Treinos realizados:", st.session_state['treinos_realizados'])

                # Identificar o tipo de problema para usar a métrica apropriada
                if st.session_state.model_type == "Classificação":
                    melhores_metricas = sorted(
                        st.session_state['treinos_realizados'], 
                        key=lambda x: x.get('Accuracy', 0),  # Usar Accuracy para classificação
                        reverse=True
                    )[0]  # Escolher o melhor modelo
                elif st.session_state.model_type == "Regressão":
                    melhores_metricas = sorted(
                        st.session_state['treinos_realizados'], 
                        key=lambda x: x.get('R²', 0),  # Usar R² para regressão
                        reverse=True
                    )[0]  # Escolher o melhor modelo

                # Seleção de modelo manual ou manter o melhor automaticamente
                model_options = [resultado['Modelo'] for resultado in st.session_state['treinos_realizados']]
                default_index = model_options.index(melhores_metricas['Modelo']) if melhores_metricas['Modelo'] in model_options else 0

                selected_model_temp = st.selectbox(
                    "Escolha um modelo para avançar para a Seleção de Features:",
                    options=model_options,
                    index=default_index
                )

                # Botão para avançar
                if st.button("Avançar para Seleção de Features"):
                    # Atualizar o modelo selecionado no session_state apenas ao clicar no botão
                    st.session_state.selected_model_name = selected_model_temp
                    st.session_state.step = 'feature_selection'
                    st.session_state['treino_concluido'] = False
                    st.rerun()
            else:
                st.error("Nenhum modelo foi treinado. Execute o treino primeiro.")


# Função para treinar e avaliar os modelos de clustering
def train_clustering_model(model, X_data, model_name):
    if model_name == "KMeans":
        model.set_params(n_clusters=st.session_state.kmeans_clusters)
        model.fit(X_data)
        st.session_state['labels'] = model.labels_

    elif model_name == "Clustering Hierárquico":
        model.set_params(n_clusters=st.session_state.kmeans_clusters, linkage="ward")
        model.fit(X_data)
        st.session_state['labels'] = model.labels_

    st.write(f"Clusterização realizada com {model_name}")
    
# Visualização dos Clusters usando PCA
def visualize_clusters(X_data):
    if 'labels' in st.session_state:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state['labels'], cmap='viridis')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Visualização dos Clusters em 2D')
        st.pyplot(plt.gcf())

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"R²": r2, "MAE": mae,"MSE": mse }

def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, use_grid_search, manual_params=None):
    metrics = {}  # Armazena as métricas

    # Verificar se é regressão
    is_regression = isinstance(model, (LinearRegression, SVR)) or "Regressão" in model.__class__.__name__

    try:
        # Recuperar parâmetros manuais do estado
        manual_params = st.session_state.get('manual_params', {})

        # Remover parâmetros inválidos antes do treino
        if manual_params.get('kernel') == 'linear' and 'gamma' in manual_params:
            del manual_params['gamma']
        if 'gamma' in st.session_state.get('manual_params', {}):
            del st.session_state['manual_params']['gamma']

        # Aplicar GridSearch se necessário
        if use_grid_search:
            # Atualizar param_grid com parâmetros manuais
            if manual_params:
                for param, value in manual_params.items():
                    if not isinstance(value, list):
                        manual_params[param] = [value]
                param_grid.update(manual_params)

            # Configurar GridSearchCV
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2' if is_regression else 'accuracy'

            grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv_strategy, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Melhor modelo e parâmetros
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Remover 'gamma' se não for necessário
            if best_params.get('kernel') == 'linear' and 'gamma' in best_params:
                del best_params['gamma']

            # Salvar no estado global
            st.session_state['manual_params'] = best_params
            st.session_state['best_params_str'] = json.dumps(best_params, indent=2)

        else:
            # Treinar sem GridSearch
            valid_params = model.get_params().keys()
            manual_params = {k: v for k, v in manual_params.items() if k in valid_params}
            # Aplicar parâmetros do GridSearch antes do treino
            best_params = st.session_state.get('manual_params', {})
            if best_params:
                model.set_params(**best_params)

            # Selecione o método de treino
            if use_grid_search:
                grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv_strategy, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.session_state['best_params_selected'] = best_params  # Salva os melhores parâmetros
            else:
                model.fit(X_train, y_train)
                best_model = model

            best_model = model
            best_params = manual_params

        # Fazer previsões
        y_pred = best_model.predict(X_test)

        # Avaliar desempenho
        if is_regression:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "Modelo": model.__class__.__name__,
                "R²": r2,
                "MAE": mae,
                "MSE": mse,
                "Best Parameters": best_params
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics = {
                "Modelo": model.__class__.__name__,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Best Parameters": best_params
            }

        return metrics

    except Exception as e:
        st.error(f"Erro ao treinar o modelo: {str(e)}")
        return None

# Função para selecionar o scoring
def select_scoring():
    # Verifica se 'selected_scoring' já existe, caso contrário, inicializa com 'f1' como padrão
    if 'selected_scoring' not in st.session_state:
        st.session_state.selected_scoring = 'F1-Score'  # Definir 'f1' como valor padrão

    # Agora o selectbox usa o valor já armazenado em 'selected_scoring'
    st.session_state.selected_scoring = st.selectbox(
        "Escolha o scoring para a seleção de features:",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        index=['Accuracy', 'Precision', 'Recall', 'F1-Score'].index(st.session_state.selected_scoring)
    )

    # Exibir a escolha armazenada
    st.write("Scoring selecionado:", st.session_state.selected_scoring)

    # Salvar em um arquivo ou variável para persistência adicional
    if st.button("Salvar escolha"):
        with open("scoring_choice.txt", "w") as file:
            file.write(st.session_state.selected_scoring)
        st.success("Escolha salva com sucesso!")


# Função para remover features correlacionadas
def remove_highly_correlated_features(df, threshold=0.9):
    """
    Remove features altamente correlacionadas.
    
    Parâmetros:
    - df: DataFrame de entrada
    - threshold: Limiar de correlação (padrão 0.9)
    
    Retorna:
    - DataFrame com features não correlacionadas
    """
    # Calcular matriz de correlação absoluta
    corr_matrix = df.corr().abs()
    
    # Obter a matriz triangular superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identificar colunas a serem removidas
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Informar quais features serão removidas (opcional)
    if to_drop:
        st.info(f"Features removidas por alta correlação: {to_drop}")
    
    # Retornar DataFrame sem as features correlacionadas
    return df.drop(columns=to_drop)


# Função para selecionar features importantes com RandomForest
def select_important_features(X, y, threshold=0.01, model_type=None):
    """
    Seleciona features importantes usando RandomForest.
    
    Parâmetros:
    - X: Matriz de features
    - y: Vetor de rótulos
    - threshold: Limiar de importância (padrão 0.01)
    - model_type: Tipo de modelo (Classificação ou Regressão)
    
    Retorna:
    - DataFrame com features importantes
    """
    # Definir o modelo baseado no tipo de problema
    if model_type == "Classificação":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Regressão":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Tipo de modelo deve ser 'Classificação' ou 'Regressão'")
    
    # Usar SimpleImputer para lidar com valores ausentes
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Treinar o modelo
    model.fit(X_imputed, y)
    
    # Calcular importância das features
    importances = model.feature_importances_
    
    # Criar DataFrame de importância das features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Selecionar features com importância acima do threshold
    important_features = feature_importance[feature_importance['importance'] > threshold]['feature']
    
    # Informar quais features foram selecionadas
    st.info(f"Features selecionadas: {list(important_features)}")
    
    return X[important_features]


# Função principal de seleção de features
def feature_selection():
    st.header("Seleção de Features")

    # Inicializar estados no primeiro carregamento
    if 'method_confirmed' not in st.session_state:
        st.session_state.method_confirmed = False
    if 'method_selection' not in st.session_state:
        st.session_state.method_selection = None
    if 'scoring_confirmed' not in st.session_state:
        st.session_state.scoring_confirmed = False

    # Seleção do método de scoring
    st.subheader("Escolha a Métrica de Scoring")
    scoring_options = {
        "Classificação": ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        "Regressão": ['R²', 'MAE', 'MSE']
    }

    # Verificar o tipo de modelo
    model_type = st.session_state.get('model_type', 'Classificação')
    
    # Selecionar scoring baseado no tipo de modelo
    selected_scoring = st.selectbox(
        "Escolha a métrica de scoring:",
        scoring_options.get(model_type, ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    )
    
    # Botão para confirmar scoring
    if st.button("Confirmar Scoring"):
        st.session_state.selected_scoring = selected_scoring
        st.session_state.scoring_confirmed = True
        st.success(f"Métrica de scoring {selected_scoring} confirmada!")

    # Só continua se o scoring for confirmado
    if st.session_state.scoring_confirmed:
        # Método de seleção de features
        st.subheader("Método de Seleção de Features")
        
        if not st.session_state.method_confirmed:
            method_selection = st.radio(
                "Escolha o método de seleção de features:", 
                ["Automático", "Manual"]
            )

            # Botão para confirmar método de seleção
            if st.button("Confirmar Método"):
                st.session_state.method_selection = method_selection
                st.session_state.method_confirmed = True
                st.success(f"Método {method_selection} confirmado!")

        # Etapa de seleção após confirmação do método
        if st.session_state.method_confirmed:
            # Variáveis de dados
            X_train = st.session_state.X_train.copy()
            X_test = st.session_state.X_test.copy()
            y_train = st.session_state.y_train.copy()
            y_test = st.session_state.y_test.copy()

            if st.session_state.method_selection == "Automático":
                st.write("### Seleção Automática de Features")
                
                # Calcular importância das features
                if st.session_state.model_type == "Classificação":
                    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Treinar o modelo de seleção
                feature_selector.fit(X_train, y_train)
                
                # Obter importâncias das features
                importances = feature_selector.feature_importances_
                feature_importances = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Exibir tabela de importâncias
                st.write("Importância das Features:")
                st.dataframe(fix_dataframe_types(feature_importances))
                
                # Threshold padrão de 1%
                threshold = 0.01
                selected_features = feature_importances[feature_importances['importance'] > threshold]['feature'].tolist()
                
                # Selecionar features
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]

            else:  # Seleção Manual
                st.write("### Seleção Manual de Features")
                
                # Calcular importância das features para exibir
                if st.session_state.model_type == "Classificação":
                    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Treinar o modelo de seleção
                feature_selector.fit(X_train, y_train)
                
                # Obter importâncias das features
                importances = feature_selector.feature_importances_
                feature_importances = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Exibir tabela de importâncias para guiar a seleção
                st.write("Importância das Features (em ordem decrescente):")
                st.dataframe(fix_dataframe_types(feature_importances))

                # Slider para número de features
                num_features = st.slider(
                    "Número de Features a Selecionar", 
                    min_value=1, 
                    max_value=X_train.shape[1], 
                    value=min(5, X_train.shape[1])
                )
                
                # Selecionar as top N features
                selected_features = feature_importances['feature'].head(num_features).tolist()
                
                # Selecionar features
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]

            # Exibir informações sobre a seleção
            st.write(f"### Features Selecionadas ({len(selected_features)})")
            st.write(selected_features)
            
            # Informações comparativas
            st.write("### Comparação de Dimensões")
            st.write(f"- Features Originais (Treino): {X_train.shape[1]}")
            st.write(f"- Features Selecionadas (Treino): {X_train_selected.shape[1]}")
            
            # Salvar no estado da sessão
            st.session_state.X_train_selected = X_train_selected
            st.session_state.X_test_selected = X_test_selected
            st.session_state.selected_features = selected_features
            st.session_state.feature_selection_done = True

            # Botão para confirmar seleção e avançar
            if st.button("Finalizar Seleção de Features"):
                st.session_state.step = 'train_and_store_metrics'
                st.rerun()
        
#Função para Treinar e Armazenar as metricas
import json

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder


# Função para treinar e armazenar métricas
def train_and_store_metrics(model, X_train, y_train, X_test, y_test, metric_type, use_grid_search=False, manual_params=None):
    try:
        # Imports necessários
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.model_selection import GridSearchCV, KFold

        # Imputar valores ausentes
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Garantir que y_train e y_test sejam válidos
        if y_train.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        else:
            y_train = y_train.fillna(y_train.mean())
            y_test = y_test.fillna(y_test.mean())

        # **RECUPERAR PARÂMETROS SALVOS**
        if metric_type == "Com Seleção":
            saved_params = st.session_state.get('best_params_selected', None) or st.session_state.get('best_params', None)
        else:
            saved_params = st.session_state.get('best_params', None)

        # **APLICAR PARÂMETROS SALVOS APENAS SE COMPATÍVEIS COM O MODELO**
        if saved_params and hasattr(model, 'get_params') and all(param in model.get_params() for param in saved_params):
            st.info(f"Aplicando parâmetros salvos ao modelo: {saved_params}")
            model.set_params(**saved_params)


        # **TREINO COM GRIDSEARCH OU DIRETO**
        if use_grid_search and metric_type == "Sem Seleção":
            param_grid = st.session_state.get('param_grid', {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            })

            # Definir estratégia de validação cruzada
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            if st.session_state.model_type == "Classificação":
                scoring = 'accuracy'
            else:
                scoring = 'r2'

            # Aplicar GridSearch
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv_strategy, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # **SALVAR PARÂMETROS NO ESTADO GLOBAL**
            st.session_state['best_params'] = best_params
            st.session_state['best_params_selected'] = best_params

        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = saved_params if saved_params else {}

        # **SALVAR MODELO TREINADO NO ESTADO GLOBAL**
        st.session_state['trained_model'] = best_model
        st.session_state['trained_model_name'] = best_model.__class__.__name__
        

        # **AVALIAR O MODELO**
        y_pred = best_model.predict(X_test)

        if st.session_state.model_type == "Classificação":
            metrics = {
                'F1-Score': f1_score(y_test, y_pred, average='weighted'),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Best Parameters': best_params
            }
        else:
            metrics = {
                'R²': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'Best Parameters': best_params
            }

        # **SALVAR MÉTRICAS NO ESTADO GLOBAL**
        if 'metrics' not in st.session_state:
            st.session_state['metrics'] = {}
        st.session_state['metrics'][metric_type] = metrics

        return metrics

    except Exception as e:
        st.error(f"Erro ao treinar o modelo: {str(e)}")
        return None



# Encontrar o modelo no dicionário (versão flexível)
def find_model_in_dict(models_dict, model_name):
    # Primeiro, tentar correspondência exata
    if model_name in models_dict:
        return models_dict[model_name]
    
    # Se não encontrar, tentar correspondência parcial
    for key in models_dict.keys():
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            return models_dict[key]
    
    return None

    # Encontrar o modelo
    model = find_model_in_dict(st.session_state.models, model_name)
    if model is None:
        st.error(f"O modelo {model_name} não foi encontrado na lista de modelos disponíveis.")
        st.write("Modelos disponíveis:", list(st.session_state.models.keys()))
        return
    
    # Encontrar o nome completo do modelo
    matched_model_name = next(
        (key for key in st.session_state.models.keys() 
         if model_name.lower() in key.lower() or key.lower() in model_name.lower()), 
        model_name
    )
    
    # Mapear o nome do modelo para sua versão curta
    mapped_model_name = model_name_map.get(matched_model_name, matched_model_name)

def evaluate_and_compare_models():
    st.title("Comparação dos Resultados do Treino dos Modelos")

    # Mapeamento de modelos
    model_name_map = {
        "Support Vector Classification (SVC)": "SVC",
        "K-Nearest Neighbors (KNN)": "KNeighborsClassifier",
        "Random Forest": "RandomForestClassifier",
        "Regressão Linear Simples (RLS)": "LinearRegression",
        "Regressão por Vetores de Suporte (SVR)": "SVR",
        "SVR": "Regressão por Vetores de Suporte (SVR)"
    }
    reverse_model_name_map = {v: k for k, v in model_name_map.items()}

    # Verificações preliminares
    if 'selected_features' not in st.session_state:
        st.error("Nenhuma feature foi selecionada. Por favor, volte à etapa de seleção de features.")
        return

    # Verificar se os modelos estão definidos
    if 'models' not in st.session_state or not st.session_state.models:
        st.error("Configuração de modelos não encontrada. Por favor, reinicie o processo de seleção de modelos.")
        return

    # Recuperar o tipo de modelo
    model_type = st.session_state.get('model_type', 'Indefinido')
    
    # Recuperar o nome do modelo selecionado
    model_name = st.session_state.get('selected_model_name')
    if not model_name:
        st.error("Nenhum modelo foi selecionado. Por favor, volte à etapa de seleção de modelos.")
        return

    # Mapear o nome do modelo para sua versão curta
    mapped_model_name = model_name_map.get(model_name, model_name)

    # Encontrar o modelo no dicionário
    model = st.session_state.models.get(model_name)
    if model is None:
        st.error(f"O modelo {model_name} não foi encontrado na lista de modelos disponíveis.")
        st.write("Modelos disponíveis:", list(st.session_state.models.keys()))
        return

    # Recuperar métricas originais e com seleção de features
    original_metrics = st.session_state.get('resultado_sem_selecao', {})
    selected_metrics = st.session_state.get('resultado_com_selecao', {})

    # Verificar se as métricas existem
    if not original_metrics:
        st.error("Não foi possível encontrar as métricas originais. Por favor, refaça o treinamento.")
        return

    # Obter dados originais
    try:
        X_train_original = st.session_state.X_train.copy()
        X_test_original = st.session_state.X_test.copy()
        y_train = st.session_state.y_train.copy()
        y_test = st.session_state.y_test.copy()
    except AttributeError:
        st.error("Dados de treino e teste não encontrados. Por favor, volte à etapa de preparação dos dados.")
        return
    
    # 1. Remover features correlacionadas
    corr_matrix = X_train_original.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    if to_drop:
        st.warning(f"⚠️ Removendo features altamente correlacionadas: {to_drop}")
        X_train_original.drop(columns=to_drop, inplace=True)
        X_test_original.drop(columns=to_drop, inplace=True)
    
    # 2. Preparar features selecionadas
    selected_features = st.session_state.selected_features
    
    # Garantir que todas as features selecionadas existem no DataFrame
    valid_features = [f for f in selected_features if f in X_train_original.columns]
    
    if not valid_features:
        st.error("Nenhuma feature selecionada é válida. Por favor, volte à etapa de seleção de features.")
        return

    # Selecionar apenas as features válidas
    X_train_selected = X_train_original[valid_features]
    X_test_selected = X_test_original[valid_features]
    
    # Normalizar com StandardScaler
    scaler = StandardScaler()
    X_train_scaled_selected = pd.DataFrame(
        scaler.fit_transform(X_train_selected), 
        columns=valid_features, 
        index=X_train_selected.index
    )
    X_test_scaled_selected = pd.DataFrame(
        scaler.transform(X_test_selected), 
        columns=valid_features, 
        index=X_test_selected.index
    )
    
    # Treinar modelo com features selecionadas
    model.fit(X_train_scaled_selected, y_train)
    y_pred_selected = model.predict(X_test_scaled_selected)
    
    # Recuperar parâmetros do modelo original e com seleção
    original_params = st.session_state.get('best_params', {})
    selected_params = st.session_state.get('best_params_selected', {})
    
    # Calcular métricas para o modelo com features selecionadas
    if model_type == "Classificação":
        selected_metrics = {
            "Modelo": mapped_model_name,
            "Accuracy": accuracy_score(y_test, y_pred_selected),
            "Precision": precision_score(y_test, y_pred_selected, average='weighted'),
            "Recall": recall_score(y_test, y_pred_selected, average='weighted'),
            "F1-Score": f1_score(y_test, y_pred_selected, average='weighted'),
            "Best Parameters": selected_params
        }
    elif model_type == "Regressão":
        selected_metrics = {
            "Modelo": mapped_model_name,
            "R²": r2_score(y_test, y_pred_selected),
            "MAE": mean_absolute_error(y_test, y_pred_selected),
            "MSE": mean_squared_error(y_test, y_pred_selected),
            "Best Parameters": selected_params
        }
    
    # Atualizar métricas com seleção de features
    st.session_state['resultado_com_selecao'] = selected_metrics
    
    # Criar DataFrame de comparação
    if model_type == "Classificação":
        comparison_df = pd.DataFrame({
            'Modelo': [original_metrics['Modelo'], selected_metrics['Modelo']],
            'Accuracy': [original_metrics.get('Accuracy', 0), selected_metrics.get('Accuracy', 0)],
            'Precision': [original_metrics.get('Precision', 0), selected_metrics.get('Precision', 0)],
            'Recall': [original_metrics.get('Recall', 0), selected_metrics.get('Recall', 0)],
            'F1-Score': [original_metrics.get('F1-Score', 0), selected_metrics.get('F1-Score', 0)],
            'Best Parameters': [original_params, selected_params]
        })
    else:
        comparison_df = pd.DataFrame({
            'Modelo': [original_metrics['Modelo'], selected_metrics['Modelo']],
            'R²': [original_metrics.get('R²', 0), selected_metrics.get('R²', 0)],
            'MAE': [original_metrics.get('MAE', 0), selected_metrics.get('MAE', 0)],
            'MSE': [original_metrics.get('MSE', 0), selected_metrics.get('MSE', 0)],
            'Best Parameters': [original_params, selected_params]
        })
    
    # Exibir tabela de comparação
    st.subheader("📈 Comparação dos Resultados:")
    st.table(fix_dataframe_types(comparison_df.style.format({
        'Accuracy': '{:.4f}', 
        'Precision': '{:.4f}', 
        'Recall': '{:.4f}', 
        'F1-Score': '{:.4f}',
        'R²': '{:.4f}',
        'MAE': '{:.4f}',
        'MSE': '{:,.4f}'
    })))
    
    # Gráfico de comparação
    metric_to_plot = 'F1-Score' if model_type == 'Classificação' else 'R²'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = comparison_df['Modelo']
    y = comparison_df[metric_to_plot]
    
    bars = ax.bar(x, y, width=0.6)
    
    # Adicionar rótulos de valor nas barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)
    
    ax.set_title(f'Comparação de {metric_to_plot}')
    ax.set_ylabel(metric_to_plot)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Botão para próxima etapa
    if st.button("Seguir para Resumo Final", key="btn_resumo_final"):
        st.session_state.step = 'final_page'
        st.rerun()

# Função para gerar interpretação personalizada das métricas
def generate_metrics_interpretation(metrics):
    interpretacao = []

    # Accuracy
    accuracy = float(metrics['Accuracy'])
    if accuracy > 0.9:
        interpretacao.append(f"- Acurácia: {accuracy:.4f} - Excelente! O modelo tem uma taxa de acerto global muito elevada.")
    elif accuracy > 0.75:
        interpretacao.append(f"- Acurácia: {accuracy:.4f} - Boa. O modelo está a funcionar bem, mas ainda há margem para otimização.")
    elif accuracy > 0.5:
        interpretacao.append(f"- Acurácia: {accuracy:.4f} - Moderada. Os erros ainda são significativos e devem ser corrigidos.")
    else:
        interpretacao.append(f"- Acurácia: {accuracy:.4f} - Fraca. O modelo está a falhar em muitas previsões e precisa de ser revisto.")

    # Precision
    precision = float(metrics['Precision'])
    if precision > 0.9:
        interpretacao.append(f"- Precisão: {precision:.4f} - Excelente! O modelo está a evitar a maioria dos falsos positivos.")
    elif precision > 0.75:
        interpretacao.append(f"- Precisão: {precision:.4f} - Bom. O modelo evita falsos positivos, mas pode ser mais rigoroso.")
    elif precision > 0.5:
        interpretacao.append(f"- Precisão: {precision:.4f} - Moderada. Há um número considerável de falsos positivos a corrigir.")
    else:
        interpretacao.append(f"- Precisão: {precision:.4f} - Fraca. Muitos falsos positivos estão a prejudicar a confiança nas previsões.")

    # Recall
    recall = float(metrics['Recall'])
    if recall > 0.9:
        interpretacao.append(f"- Recall: {recall:.4f} - Excelente! O modelo está a identificar quase todos os positivos verdadeiros.")
    elif recall > 0.75:
        interpretacao.append(f"- Recall: {recall:.4f} - Bom. A maioria dos positivos verdadeiros é identificada, mas há espaço para melhorias.")
    elif recall > 0.5:
        interpretacao.append(f"- Recall: {recall:.4f} - Moderado. O modelo está a perder demasiados positivos verdadeiros.")
    else:
        interpretacao.append(f"- Recall: {recall:.4f} - Fraco. O modelo falha em identificar a maioria dos positivos verdadeiros. Pode ser necessário ajustar os pesos ou thresholds.")
    
    # F1-Score
    f1_score = float(metrics['F1-Score'])
    if f1_score > 0.9:
        interpretacao.append(f"- F1-Score: {f1_score:.4f} - Excelente equilíbrio entre precisão e sensibilidade. O modelo está altamente otimizado.")
    elif f1_score > 0.75:
        interpretacao.append(f"- F1-Score: {f1_score:.4f} - Bom desempenho. Contudo, há espaço para melhorias nos falsos positivos ou negativos.")
    elif f1_score > 0.5:
        interpretacao.append(f"- F1-Score: {f1_score:.4f} - Desempenho moderado. Ajustes no treino ou balanceamento dos dados podem ajudar.")
    else:
        interpretacao.append(f"- F1-Score: {f1_score:.4f} - Desempenho fraco. Recomenda-se rever os dados, ajustar hiperparâmetros ou otimizar o modelo.")


    # Conclusão Geral
    if f1_score > 0.9 and precision > 0.9 and recall > 0.9:
        interpretacao.append("\nConclusão Geral: 🎉 O modelo apresenta um desempenho excecional em todas as métricas. Está pronto para produção!")
    elif f1_score > 0.75 and precision > 0.75 and recall > 0.75:
        interpretacao.append("\nConclusão Geral: 👍 O modelo tem um bom desempenho geral, mas pode ser ligeiramente melhorado com ajustes finos.")
    elif f1_score > 0.5 or precision > 0.5 or recall > 0.5:
        interpretacao.append("\nConclusão Geral:⚠️ O modelo tem um desempenho moderado. Recomenda-se ajustar os hiperparâmetros ou melhorar os dados de treino.")
    else:
        interpretacao.append("\nConclusão Geral: ❗ O modelo apresenta um desempenho fraco. Será necessário rever o processo de treino, os dados e os parâmetros.")

    return "\n".join(interpretacao)

def generate_regression_interpretation(metrics):
    interpretation = []

    # R² (Coeficiente de Determinação)
    r2 = float(metrics['R²'])
    if r2 > 0.9:
        interpretation.append(f"- R²: {r2:.4f} - Excelente! O modelo explica quase toda a variabilidade dos dados. Isso indica um forte ajuste entre as previsões e os valores reais.")
    elif r2 > 0.75:
        interpretation.append(f"- R²: {r2:.4f} - Muito bom! O modelo explica a maior parte da variabilidade dos dados, mas ainda pode ser melhorado.")
    elif r2 > 0.5:
        interpretation.append(f"- R²: {r2:.4f} - Moderado. O modelo consegue explicar uma parte significativa da variabilidade, mas há limitações importantes no ajuste.")
    else:
        interpretation.append(f"- R²: {r2:.4f} - Fraco. O modelo explica pouca variabilidade dos dados. Considere revisar as features ou usar um modelo mais adequado.")

    # MAE (Erro Absoluto Médio)
    mae = float(metrics['MAE'])
    if mae < 0.1:
        interpretation.append(f"- MAE: {mae:.4f} - Excelente! O erro absoluto médio é muito pequeno, sugerindo que as previsões são altamente precisas.")
    elif mae < 1:
        interpretation.append(f"- MAE: {mae:.4f} - Bom. O erro absoluto médio é aceitável, mas ainda pode ser otimizado.")
    else:
        interpretation.append(f"- MAE: {mae:.4f} - Alto. As previsões estão frequentemente desviando dos valores reais. Considere ajustar o modelo ou as features.")

    # MSE (Erro Quadrático Médio)
    mse = float(metrics['MSE'])
    if mse < 0.1:
        interpretation.append(f"- MSE: {mse:.4f} - Excelente! O erro quadrático médio é muito baixo, indicando que as previsões estão próximas dos valores reais.")
    elif mse < 1:
        interpretation.append(f"- MSE: {mse:.4f} - Bom. O erro é relativamente baixo, mas ainda há espaço para reduzir as discrepâncias.")
    else:
        interpretation.append(f"- MSE: {mse:.4f} - Alto. O erro é significativo. Isso pode indicar que o modelo não está capturando bem os padrões nos dados.")

    # Conclusão geral com base nas métricas
    if r2 > 0.9 and mse < 0.1 and mae < 0.1:
        interpretation.append("\nConclusão Geral: 🎉 O modelo apresenta um desempenho excepcional! Está pronto para produção.")
    elif r2 > 0.75 and mse < 1 and mae < 1:
        interpretation.append("\nConclusão Geral: 👍 O modelo tem um bom desempenho geral. Com ajustes menores, pode se tornar ainda melhor.")
    elif r2 > 0.5 or mse < 1 or mae < 1:
        interpretation.append("\nConclusão Geral: ⚠️ O modelo está funcional, mas ainda apresenta limitações. Ajustes adicionais são recomendados.")
    else:
        interpretation.append("\nConclusão Geral: ❗ O modelo apresenta desempenho insatisfatório. Considere reavaliar as features, ajustar hiperparâmetros ou explorar modelos alternativos.")

    return "\n".join(interpretation)

# Função para salvar o modelo treinado com nome dinâmico
def save_best_model(model, with_feature_selection=True):
    try:
        # Determinar o nome do arquivo com base na seleção de features
        if with_feature_selection:
            model_filename = "best_model_com_selecao_features.pkl"
        else:
            model_filename = "best_model_sem_selecao_features.pkl"

        # Salvar o modelo usando joblib
        joblib.dump(model, model_filename)
        st.success(f"Modelo salvo com sucesso como {model_filename}")
        return model_filename
    except Exception as e:
        st.error(f"Erro ao salvar o modelo: {str(e)}")
        return None


def execute_training():
    if st.session_state.step == 'train_and_store_metrics':
        model = st.session_state.models[st.session_state.selected_model_name]

        metrics = train_and_store_metrics(
            model,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test,
            metric_type="sem_selecao_features"
        )

        # Depuração
        st.write("Conteúdo de metrics após treino:", st.session_state.get('metrics', {}))

        # Avançar para a página final
        st.session_state.step = 'final_page'
        st.rerun()


## Relatório Final para Classificação/Regressao ##

# Função para gerar o relatório em PDF
class CustomPDF(FPDF):
    def header(self):
        # Baixar a imagem do logo e salvar localmente
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        response = requests.get(logo_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                tmpfile.write(response.content)
                tmpfile_path = tmpfile.name
                # Adicionar a imagem no cabeçalho
                self.image(tmpfile_path, 10, 8, 20) 
        else:
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, "Logo não disponível", align='C')
        
        # Definir fonte para o cabeçalho
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', align='C', ln=True)
        self.ln(15)  # Espaço após o cabeçalho

    def footer(self):
        # Ir para 1.5 cm da parte inferior
        self.set_y(-15)
        # Definir fonte para o rodapé
        self.set_font('Helvetica', 'I', 10)
        # Data atual
        current_date = datetime.now().strftime('%d/%m/%Y')
        # Adicionar rodapé com a data e número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', align='C')



class MLCaseModelReportGenerator:
    def __init__(self, output_path='model_performance_report.pdf', logo_url=None):
        """
        Initialize the report generator
        
        :param output_path: Path to save the PDF
        :param logo_url: Optional URL for organization logo
        """
        self.output_path = output_path
        self.logo_url = logo_url or 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        
        # Fetch logo
        self.logo_path = self._fetch_logo()
        
        # Prepare styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _fetch_logo(self):
        """Fetch and save logo image"""
        try:
            response = requests.get(self.logo_url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    tmpfile.write(response.content)
                    return tmpfile.name
            return None
        except Exception:
            return None
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='MLCaseTitle',
            parent=self.styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#2C3E50'),
            alignment=1,  # Center alignment
            spaceAfter=12
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='MLCaseSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=6
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='MLCaseNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2C3E50'),
            leading=14
        ))
    
    def create_bar_chart(self, data, labels, title):
        """Create a bar chart using matplotlib and return as an image buffer"""
        plt.figure(figsize=(6, 4), dpi=100)
        plt.bar(labels, data, color=['#3498DB', '#2980B9'])
        plt.title(title, fontsize=12, color='#2C3E50')
        plt.ylabel('Value', color='#2C3E50')
        plt.xticks(rotation=45, ha='right', color='#2C3E50')
        plt.tight_layout()
        
        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
def gerar_relatorio_pdf(comparison_df, best_model, session_state):
    """
    Gera um relatório PDF com os resultados da comparação de modelos.
    
    Args:
        comparison_df: DataFrame com as métricas comparativas
        best_model: String com o nome do melhor modelo
        session_state: Estado da sessão do Streamlit
        
    Returns:
        BytesIO: Buffer contendo o PDF gerado
    """
    # Importar bibliotecas necessárias
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    # Função para limpar texto para compatibilidade com codificação Latin-1
    def clean_text(text):
        if not isinstance(text, str):
            return str(text)
        return text.encode('latin-1', errors='ignore').decode('latin-1')
    
    # Inicialização do PDF
    pdf = CustomPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Título do Relatório
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, txt=clean_text("Relatório Final dos Modelos Treinados"), ln=True, align="C")
    pdf.ln(10)
    
    # Tipo de Modelo
    model_type = session_state.get('model_type', 'Indefinido')
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Tipo de Modelo:"), ln=False)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt=clean_text(model_type), ln=True)
    
    # Modelo Selecionado
    selected_model_name = session_state.get('selected_model_name', 'Não Selecionado')
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Modelo Selecionado:"), ln=False)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt=clean_text(selected_model_name), ln=True)
    
    # Melhor Modelo
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Melhor Modelo:"), ln=False)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt=clean_text(best_model), ln=True)
    pdf.ln(10)
    
    # Informações do Conjunto de Dados
    if 'X_train' in session_state and 'X_test' in session_state:
        X_train = session_state.X_train
        X_test = session_state.X_test
        
        # Calcular percentuais e tamanhos
        total_samples = X_train.shape[0] + X_test.shape[0]
        train_percent = (X_train.shape[0] / total_samples) * 100
        test_percent = (X_test.shape[0] / total_samples) * 100
        
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(0, 10, txt=clean_text("Informações dos Conjuntos de Dados"), ln=True)
        pdf.ln(5)
        
        # Tabela de informações do conjunto de dados
        data_info = [
            ["Amostras de Treino", f"{X_train.shape[0]} ({train_percent:.1f}%)"],
            ["Amostras de Teste", f"{X_test.shape[0]} ({test_percent:.1f}%)"],
            ["Features Originais", f"{X_train.shape[1]}"]
        ]
        
        # Adicionar features após seleção se estiverem disponíveis
        if 'X_train_selected' in session_state:
            data_info.append(["Features Após Seleção", f"{session_state.X_train_selected.shape[1]}"])
        
        # Formatar a tabela de informações
        pdf.set_font("Helvetica", size=10)
        pdf.set_fill_color(200, 220, 255)  # Cor de fundo do cabeçalho
        
        for i, (label, value) in enumerate(data_info):
            if i % 2 == 0:  # Linhas alternadas
                pdf.set_fill_color(240, 240, 240)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(70, 8, txt=clean_text(label), border=1, ln=0, fill=True)
            pdf.cell(0, 8, txt=clean_text(value), border=1, ln=1, fill=True)
        
        pdf.ln(10)
    
    # Features Selecionadas
    if 'selected_features' in session_state:
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(0, 10, txt=clean_text("Features Selecionadas"), ln=True)
        
        # Listar as features
        features = session_state.selected_features
        pdf.set_font("Helvetica", size=10)
        for i, feature in enumerate(features):
            pdf.cell(0, 6, txt=clean_text(f"• {feature}"), ln=True)
        
        pdf.ln(10)
    
    # Tabela de Métricas
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Comparação de Métricas"), ln=True)
    
    # Verificar o tipo de modelo para determinar quais métricas exibir
    is_regression = model_type == "Regressão"
    metric_columns = ['R²', 'MAE', 'MSE'] if is_regression else ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Criar tabela de métricas
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.set_fill_color(200, 220, 255)  # Cor de fundo do cabeçalho
    
    # Definir a largura das colunas
    column_width = 30
    first_column_width = 60
    
    # Cabeçalho da tabela
    pdf.cell(first_column_width, 10, "Modelo", 1, 0, 'C', True)
    for col in metric_columns:
        pdf.cell(column_width, 10, clean_text(col), 1, 0, 'C', True)
    pdf.ln()
    
    # Linhas da tabela
    pdf.set_font("Helvetica", size=10)
    for _, row in comparison_df.iterrows():
        model_name = row['Modelo']
        pdf.cell(first_column_width, 10, clean_text(model_name), 1, 0, 'L')
        
        for col in metric_columns:
            if col in row:
                # Formatar o valor numérico com 4 casas decimais
                if isinstance(row[col], (int, float)):
                    value = f"{row[col]:.4f}"
                else:
                    value = str(row[col])
                pdf.cell(column_width, 10, clean_text(value), 1, 0, 'C')
        
        pdf.ln()
    
    pdf.ln(10)
    
    # Gráficos de Métricas
    # Para cada métrica, criar um gráfico de barras
    for metric in metric_columns:
        if metric in comparison_df.columns:
            # Criar o gráfico
            plt.figure(figsize=(8, 4))
            
            # Dados para o gráfico
            models = comparison_df['Modelo'].tolist()
            values = comparison_df[metric].tolist()
            
            # Criar barras
            plt.bar(models, values, color=['#3498DB', '#2980B9'])
            
            # Adicionar valores sobre as barras
            for i, v in enumerate(values):
                if isinstance(v, (int, float)):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
            
            # Estilização
            plt.title(f"Comparação de {metric}")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Salvar o gráfico em um arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name)
            plt.close()
            
            # Adicionar o gráfico ao PDF
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 10, txt=clean_text(f"Gráfico de Comparação - {metric}"), ln=True, align="C")
            pdf.image(temp_file.name, x=10, y=40, w=190)
            
            # Fechar e remover o arquivo temporário
            temp_file.close()
    
    # Interpretação das Métricas
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Interpretação das Métricas"), ln=True)
    
    # Função para gerar interpretação de métricas
    def generate_metrics_interpretation(metrics, model_type):
        interpretacao = []
        
        if model_type == "Classificação":
            # Accuracy
            accuracy = float(metrics.get('Accuracy', 0))
            if accuracy > 0.9:
                interpretacao.append(f"Acurácia: {accuracy:.4f} - Excelente! O modelo tem uma taxa de acerto global muito elevada.")
            elif accuracy > 0.75:
                interpretacao.append(f"Acurácia: {accuracy:.4f} - Boa. O modelo está a funcionar bem, mas ainda há margem para otimização.")
            elif accuracy > 0.5:
                interpretacao.append(f"Acurácia: {accuracy:.4f} - Moderada. Os erros ainda são significativos e devem ser corrigidos.")
            else:
                interpretacao.append(f"Acurácia: {accuracy:.4f} - Fraca. O modelo está a falhar em muitas previsões e precisa de ser revisto.")
        
            # Precision
            precision = float(metrics.get('Precision', 0))
            if precision > 0.9:
                interpretacao.append(f"Precisão: {precision:.4f} - Excelente! O modelo está a evitar a maioria dos falsos positivos.")
            elif precision > 0.75:
                interpretacao.append(f"Precisão: {precision:.4f} - Bom. O modelo evita falsos positivos, mas pode ser mais rigoroso.")
            elif precision > 0.5:
                interpretacao.append(f"Precisão: {precision:.4f} - Moderada. Há um número considerável de falsos positivos a corrigir.")
            else:
                interpretacao.append(f"Precisão: {precision:.4f} - Fraca. Muitos falsos positivos estão a prejudicar a confiança nas previsões.")
        
            # Recall
            recall = float(metrics.get('Recall', 0))
            if recall > 0.9:
                interpretacao.append(f"Recall: {recall:.4f} - Excelente! O modelo está a identificar quase todos os positivos verdadeiros.")
            elif recall > 0.75:
                interpretacao.append(f"Recall: {recall:.4f} - Bom. A maioria dos positivos verdadeiros é identificada, mas há espaço para melhorias.")
            elif recall > 0.5:
                interpretacao.append(f"Recall: {recall:.4f} - Moderado. O modelo está a perder demasiados positivos verdadeiros.")
            else:
                interpretacao.append(f"Recall: {recall:.4f} - Fraco. O modelo falha em identificar a maioria dos positivos verdadeiros.")
            
            # F1-Score
            f1_score = float(metrics.get('F1-Score', 0))
            if f1_score > 0.9:
                interpretacao.append(f"F1-Score: {f1_score:.4f} - Excelente equilíbrio entre precisão e sensibilidade.")
            elif f1_score > 0.75:
                interpretacao.append(f"F1-Score: {f1_score:.4f} - Bom desempenho. Contudo, há espaço para melhorias.")
            elif f1_score > 0.5:
                interpretacao.append(f"F1-Score: {f1_score:.4f} - Desempenho moderado.")
            else:
                interpretacao.append(f"F1-Score: {f1_score:.4f} - Desempenho fraco.")
        
        elif model_type == "Regressão":
            # R² (Coeficiente de Determinação)
            r2 = float(metrics.get('R²', 0))
            if r2 > 0.9:
                interpretacao.append(f"R²: {r2:.4f} - Excelente! O modelo explica quase toda a variabilidade dos dados.")
            elif r2 > 0.75:
                interpretacao.append(f"R²: {r2:.4f} - Muito bom! O modelo explica a maior parte da variabilidade dos dados.")
            elif r2 > 0.5:
                interpretacao.append(f"R²: {r2:.4f} - Moderado. O modelo consegue explicar uma parte significativa da variabilidade.")
            else:
                interpretacao.append(f"R²: {r2:.4f} - Fraco. O modelo explica pouca variabilidade dos dados.")
        
            # MAE (Erro Absoluto Médio)
            mae = float(metrics.get('MAE', 0))
            if mae < 0.1:
                interpretacao.append(f"MAE: {mae:.4f} - Excelente! O erro absoluto médio é muito pequeno.")
            elif mae < 1:
                interpretacao.append(f"MAE: {mae:.4f} - Bom. O erro absoluto médio é aceitável.")
            else:
                interpretacao.append(f"MAE: {mae:.4f} - Alto. As previsões estão frequentemente desviando dos valores reais.")
        
            # MSE (Erro Quadrático Médio)
            mse = float(metrics.get('MSE', 0))
            if mse < 0.1:
                interpretacao.append(f"MSE: {mse:.4f} - Excelente! O erro quadrático médio é muito baixo.")
            elif mse < 1:
                interpretacao.append(f"MSE: {mse:.4f} - Bom. O erro é relativamente baixo.")
            else:
                interpretacao.append(f"MSE: {mse:.4f} - Alto. O erro é significativo.")
        
        return interpretacao
    
    # Obter dados das métricas originais e selecionadas
    original_metrics = {}
    selected_metrics = {}
    
    # Separar as métricas por tipo de modelo
    for _, row in comparison_df.iterrows():
        model_name = row['Modelo']
        
        if "Sem Seleção" in model_name:
            # Extrair métricas do modelo sem seleção de features
            for col in metric_columns:
                if col in row:
                    original_metrics[col] = row[col]
        
        if "Com Seleção" in model_name:
            # Extrair métricas do modelo com seleção de features
            for col in metric_columns:
                if col in row:
                    selected_metrics[col] = row[col]
    
    # Interpretações para modelos sem e com seleção de features
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Modelo Sem Seleção de Features"), ln=True)
    pdf.set_font("Helvetica", size=10)
    
    # Adicionar interpretação do modelo sem seleção
    for line in generate_metrics_interpretation(original_metrics, model_type):
        pdf.multi_cell(0, 8, txt=clean_text(f"• {line}"))
    
    pdf.ln(5)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Modelo Com Seleção de Features"), ln=True)
    pdf.set_font("Helvetica", size=10)
    
    # Adicionar interpretação do modelo com seleção
    for line in generate_metrics_interpretation(selected_metrics, model_type):
        pdf.multi_cell(0, 8, txt=clean_text(f"• {line}"))
    
    # Conclusão
    pdf.ln(10)
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Conclusão"), ln=True)
    
    # Determinar a melhor métrica com base no tipo de modelo
    main_metric = 'R²' if is_regression else 'F1-Score'
    original_value = original_metrics.get(main_metric, 0)
    selected_value = selected_metrics.get(main_metric, 0)
    
    # Texto da conclusão
    pdf.set_font("Helvetica", size=10)
    conclusion_text = f"Com base na métrica principal ({main_metric}), o modelo {best_model} apresentou o melhor desempenho."
    pdf.multi_cell(0, 8, txt=clean_text(conclusion_text))
    
    if original_value > selected_value:
        recommendation_text = "Recomenda-se utilizar o modelo sem seleção de features, pois apresentou melhor desempenho geral."
    else:
        feature_reduction = session_state.X_train.shape[1] - session_state.X_train_selected.shape[1]
        recommendation_text = f"Recomenda-se utilizar o modelo com seleção de features, pois além de melhorar o desempenho, reduziu a dimensionalidade em {feature_reduction} features."
    
    pdf.multi_cell(0, 8, txt=clean_text(recommendation_text))
    
    # Salvar o PDF em um buffer
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1', errors='ignore')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Função para exibir a página final com o relatório

# Mapeamento de nomes de métricas para as colunas do DataFrame
# Atualizar o dicionário METRIC_MAPPING para garantir que MAE seja reconhecido
METRIC_MAPPING = {
    "accuracy": "Accuracy",
    "precision": "Precision", 
    "recall": "Recall",
    "f1-score": "F1-Score",
    "r2": "R²",
    "R²": "R²",  # Adicionar mapeamento direto para R²
    "r-squared": "R²",
    "coefficient_of_determination": "R²",
    "mean_squared_error": "MSE",
    "mse": "MSE",  # Adicionar versão minúscula de MSE
    "mean_absolute_error": "MAE",
    "mae": "MAE"  # Adicionar versão minúscula de MAE
}

def get_metric_mapping(metric):
    """
    Função para obter o nome da métrica de forma mais flexível
    
    Args:
        metric (str): Nome da métrica a ser mapeada
    
    Returns:
        str: Nome da métrica mapeado ou None se não encontrado
    """
    # Garantir que seja uma string
    if not isinstance(metric, str):
        st.write(f"Metric não é uma string: {metric}, tipo: {type(metric)}")
        return None
    
    # Converter para minúsculas, remover espaços, acentos
    import unidecode
    metric_clean = unidecode.unidecode(metric.lower().replace(' ', '').replace('-', '').replace('_', ''))
    
    # Verificar se a métrica já está diretamente no formato esperado
    if metric in METRIC_MAPPING.values():
        return metric
    
    # Dicionário expandido de mapeamentos
    extended_mapping = {
        **METRIC_MAPPING,
        "r2score": "R²",
        "rsquared": "R²",
        "determinacao": "R²",
        "coeficienteajuste": "R²",
        "mae": "MAE",
        "erro_absoluto_medio": "MAE",
        "mean_absolute_error": "MAE",
        "erro_absoluto": "MAE",
        "mse": "MSE",
        "erro_quadratico_medio": "MSE",
        "mean_squared_error": "MSE",
        "erro_quadratico": "MSE"
    }
    
    # Tentar mapear
    mapped_metric = extended_mapping.get(metric_clean)
    
    # Se não encontrou, verificar diretamente nas chaves do METRIC_MAPPING
    if mapped_metric is None and metric in METRIC_MAPPING:
        mapped_metric = METRIC_MAPPING[metric]
        
    # Adicionar debug
    print(f"Métrica original: {metric}, limpa: {metric_clean}, mapeada: {mapped_metric}")
    
    return mapped_metric
    
def final_page():
    st.title("Resumo Final dos Modelos Treinados")

    # **CONFIGURAÇÕES UTILIZADAS**
    st.subheader("Configurações Utilizadas")

    # Tipo de Modelo
    model_type = st.session_state.get('model_type', 'Indefinido')
    st.write(f"**Tipo de Modelo:** {model_type}")

    # Modelo Selecionado
    selected_model_name = st.session_state.get('selected_model_name', 'Não Selecionado')
    st.write(f"**Modelo Selecionado:** {selected_model_name}")

    # Recupera métricas salvas (sem re-treinar)
    original_metrics = st.session_state.get('resultado_sem_selecao', {})
    selected_metrics = st.session_state.get('resultado_com_selecao', {})

    # Exibir estatísticas sobre os conjuntos de dados
    if 'X_train' in st.session_state and 'X_train_selected' in st.session_state:
        X_train_original = st.session_state.X_train
        X_train_selected = st.session_state.X_train_selected
        
        # Calcular percentuais
        total_samples = X_train_original.shape[0] + st.session_state.X_test.shape[0]
        train_percent = (X_train_original.shape[0] / total_samples) * 100
        test_percent = (st.session_state.X_test.shape[0] / total_samples) * 100
        
        st.subheader("📊 Informações dos Conjuntos de Dados")
        st.write(f"• Amostras de Treino: {X_train_original.shape[0]} ({train_percent:.1f}% do total)")
        st.write(f"• Amostras de Teste: {st.session_state.X_test.shape[0]} ({test_percent:.1f}% do total)")
        st.write(f"• Features Originais: {st.session_state.X_train_original.shape[1] if 'X_train_original' in st.session_state else X_train_original.shape[1]}")
        st.write(f"• Features Após Seleção: {X_train_selected.shape[1]}")

    # Recuperar features selecionadas
    if 'selected_features' in st.session_state:
        st.subheader("✅ Features Selecionadas")
        st.write(st.session_state.selected_features)

    # Recupera a métrica escolhida para seleção de features
    scoring_metric = st.session_state.get("selected_scoring", None)

    # Validar se a métrica foi definida
    if not scoring_metric:
        st.error("Nenhuma métrica foi selecionada. Volte para a etapa de Seleção de Features.")
        return

    # Obter o nome capitalizado da métrica com base no mapeamento
    scoring_metric_capitalized = get_metric_mapping(scoring_metric)
    if not scoring_metric_capitalized:
        st.error(f"A métrica '{scoring_metric}' não é válida ou não está disponível.")
        
        
        return
    # **COMPARAÇÃO DE MÉTRICAS**
    st.subheader("Comparação de Métricas")

    # Formatar valores com 4 casas decimais
    def format_metric(value):
        try:
            return float(f"{float(value):.4f}")
        except (ValueError, TypeError):
            return None

    # Criar tabela de métricas
    if model_type == "Classificação":
        comparison_df = pd.DataFrame({
            'Modelo': ['Sem Seleção de Features', 'Com Seleção de Features'],
            'Accuracy': [
                format_metric(original_metrics.get('Accuracy', 'N/A')),
                format_metric(selected_metrics.get('Accuracy', 'N/A'))
            ],
            'Precision': [
                format_metric(original_metrics.get('Precision', 'N/A')),
                format_metric(selected_metrics.get('Precision', 'N/A'))
            ],
            'Recall': [
                format_metric(original_metrics.get('Recall', 'N/A')),
                format_metric(selected_metrics.get('Recall', 'N/A'))
            ],
            'F1-Score': [
                format_metric(original_metrics.get('F1-Score', 'N/A')),
                format_metric(selected_metrics.get('F1-Score', 'N/A'))
            ],
            'Best Parameters': [
                st.session_state.get('best_params', 'N/A'),
                st.session_state.get('best_params_selected', 'N/A')
            ]
        })
    elif model_type == "Regressão":
        comparison_df = pd.DataFrame({
            'Modelo': ['Sem Seleção de Features', 'Com Seleção de Features'],
            'R²': [
                format_metric(original_metrics.get('R²', 'N/A')),
                format_metric(selected_metrics.get('R²', 'N/A'))
            ],
            'MAE': [
                format_metric(original_metrics.get('MAE', 'N/A')),
                format_metric(selected_metrics.get('MAE', 'N/A'))
            ],
            'MSE': [
                format_metric(original_metrics.get('MSE', 'N/A')),
                format_metric(selected_metrics.get('MSE', 'N/A'))
            ],
            'Best Parameters': [
                st.session_state.get('best_params', 'N/A'),
                st.session_state.get('best_params_selected', 'N/A')
            ]
        })
    else:
        st.error("Tipo de modelo não reconhecido. Não é possível gerar a tabela de métricas.")
        return

    # Exibir tabela de métricas
    st.table(fix_dataframe_types(comparison_df.style.format({
        'R²': '{:.4f}' if 'R²' in comparison_df.columns else None,
        'MAE': '{:.4f}' if 'MAE' in comparison_df.columns else None,
        'MSE': '{:,.4f}' if 'MSE' in comparison_df.columns else None,
        'Accuracy': '{:.4f}' if 'Accuracy' in comparison_df.columns else None,
        'Precision': '{:.4f}' if 'Precision' in comparison_df.columns else None,
        'Recall': '{:.4f}' if 'Recall' in comparison_df.columns else None,
        'F1-Score': '{:.4f}' if 'F1-Score' in comparison_df.columns else None
    })))

    # Verificar se a métrica escolhida existe no DataFrame
    if scoring_metric_capitalized not in comparison_df.columns:
        st.error(f"A métrica '{scoring_metric}' não está disponível no DataFrame.")
        return

    # **GRÁFICOS DAS MÉTRICAS**
    st.subheader("Gráfico Interativo de Comparação de Métricas")

    # Determinar as métricas disponíveis com base no tipo de modelo
    if model_type == "Classificação":
        metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    elif model_type == "Regressão":
        metric_columns = ['R²', 'MAE', 'MSE']
    else:
        st.error("Tipo de modelo não reconhecido. Não é possível gerar gráficos.")
        return

    # Adicionar um filtro interativo para a seleção da métrica
    selected_metric = st.selectbox(
        "Selecione a métrica para visualizar:",
        metric_columns,
        index=0  # Métrica padrão exibida no início
    )

    # Criar o gráfico apenas para a métrica selecionada
    if selected_metric in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Dados para o gráfico
        bars = ax.bar(
            comparison_df['Modelo'],
            comparison_df[selected_metric]
        )
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12)
        
        ax.set_title(f"Comparação de {selected_metric}", fontsize=14)
        ax.set_ylabel(selected_metric, fontsize=12)
        ax.set_xlabel("Modelo", fontsize=12)
        
        # Ajustar altura para caber os valores
        plt.ylim(0, max(comparison_df[selected_metric]) * 1.1)
        
        # Exibir gráfico no Streamlit
        st.pyplot(fig)
    else:
        st.error(f"A métrica selecionada '{selected_metric}' não está disponível.")

    # **DETERMINAR O MELHOR MODELO BASEADO NA MÉTRICA ESCOLHIDA**
    scoring_values = comparison_df[scoring_metric_capitalized].values  # Recupera os valores da métrica na tabela
    if len(scoring_values) == 2:  # Certifique-se de que existem dois valores (sem e com seleção)
        score_without_selection = scoring_values[0]
        score_with_selection = scoring_values[1]

        # Determina o melhor modelo
        if score_with_selection is not None and score_without_selection is not None:
            if score_with_selection > score_without_selection:
                best_model = "Com Seleção de Features"
                best_score = score_with_selection
            else:
                best_model = "Sem Seleção de Features"
                best_score = score_without_selection
        elif score_with_selection is not None:
            best_model = "Com Seleção de Features"
            best_score = score_with_selection
        elif score_without_selection is not None:
            best_model = "Sem Seleção de Features"
            best_score = score_without_selection
        else:
            st.warning("Não foi possível determinar o melhor modelo.")
            return
    else:
        st.warning("Erro na determinação das métricas na tabela.")
        return

    # Exibir mensagem com o melhor modelo
    st.success(f"🎉 **O melhor modelo é:** {best_model} com base na métrica: {scoring_metric_capitalized} ({best_score:.4f})")

    # **INTERPRETAÇÃO DAS MÉTRICAS**
    st.subheader("Interpretação das Métricas")
    try:
        # Gerar interpretação para cada modelo
        if model_type == "Classificação":
            interpretation_without = generate_metrics_interpretation(original_metrics)
            interpretation_with = generate_metrics_interpretation(selected_metrics)
        elif model_type == "Regressão":
            interpretation_without = generate_regression_interpretation(original_metrics)
            interpretation_with = generate_regression_interpretation(selected_metrics)
        else:
            raise ValueError("Tipo de modelo desconhecido para interpretação.")

        # Exibir interpretações
        st.write("### Sem Seleção de Features")
        st.write(interpretation_without)

        st.write("### Com Seleção de Features")
        st.write(interpretation_with)
    except Exception as e:
        st.error(f"Erro ao gerar a interpretação das métricas: {e}")
        
    # **DOWNLOAD DO MODELO TREINADO**
    st.subheader("Download do Melhor Modelo Treinado")
    model = st.session_state.models.get(st.session_state.selected_model_name)
    if best_model == "Com Seleção de Features":
        model_filename = save_best_model(model, with_feature_selection=True)
    else:
        model_filename = save_best_model(model, with_feature_selection=False)

    if model_filename:
        with open(model_filename, "rb") as file:
            st.download_button(
                label="💾 Download Melhor Modelo",
                data=file,
                file_name=model_filename,
                mime="application/octet-stream",
            )

    # **DOWNLOAD DO RELATÓRIO EM PDF**
    try:
        pdf_buffer = gerar_relatorio_pdf(comparison_df, best_model, st.session_state)
        # Recuperar o nome do modelo selecionado
        selected_model_name = st.session_state.get('selected_model_name', 'modelo')
        # Criar o nome do arquivo incluindo o modelo
        pdf_file_name = f"relatorio_final_{selected_model_name}.pdf"
        # Adicionar o botão de download com o nome personalizado
        st.download_button(
            label="💾 Download Relatório PDF",
            data=pdf_buffer,
            file_name=pdf_file_name,
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"Erro ao gerar relatório em PDF: {e}")

    # **CONCLUIR**
    if st.button("Concluir"):
        st.session_state.clear()  # Limpa o cache do Streamlit
        st.rerun()
        
############ Relatório Final para Clustering ###################
# Classe personalizada para PDF
class CustomPDF(FPDF):
    def header(self):
        # Baixar a imagem do logo e salvar localmente
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        response = requests.get(logo_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                tmpfile.write(response.content)
                tmpfile_path = tmpfile.name
                # Adicionar a imagem no cabeçalho
                self.image(tmpfile_path, 10, 8, 20) 
        else:
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, "Logo não disponível", align='C')
        
        # Definir fonte para o cabeçalho
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', align='C', ln=True)
        self.ln(15)  # Espaço após o cabeçalho

    def footer(self):
        # Ir para 1.5 cm da parte inferior
        self.set_y(-15)
        # Definir fonte para o rodapé
        self.set_font('Helvetica', 'I', 10)
        # Data atual
        current_date = datetime.now().strftime('%d/%m/%Y')
        # Adicionar rodapé com a data e número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', align='C')


# Função para gerar o relatório PDF
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

class CustomPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Relatório de Clustering", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

def gerar_relatorio_clustering_pdf(initial_metrics, retrain_metrics, best_model_type, st_session):
    pdf = CustomPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, txt="Relatório Final de Clustering", ln=True, align="C")
    pdf.ln(10)

    # Modelo Selecionado
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(50, 10, txt="Modelo Selecionado:", ln=False)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt=st_session['selected_model_name'], ln=True)
    pdf.ln(5)

    # Melhor Modelo
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(50, 10, txt="Melhor Modelo Treinado:", ln=False)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt=best_model_type, ln=True)
    pdf.ln(10)

    # Adicionar métricas do treino inicial e re-treino
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt="Métricas Obtidas", ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.set_fill_color(200, 220, 255)

    # Cabeçalho da tabela
    col_widths = [40, 40, 40, 40]
    pdf.cell(col_widths[0], 10, "Treino", 1, 0, 'C', True)
    pdf.cell(col_widths[1], 10, "Silhouette Score", 1, 0, 'C', True)
    pdf.cell(col_widths[2], 10, "Davies-Bouldin Index", 1, 0, 'C', True)
    pdf.cell(col_widths[3], 10, "Calinski-Harabasz Score", 1, 1, 'C', True)

    # Dados da tabela
    pdf.set_font("Helvetica", size=8)
    pdf.cell(col_widths[0], 10, "Treino Inicial", 1, 0, 'C')
    pdf.cell(col_widths[1], 10, f"{initial_metrics['Silhouette Score']:.2f}", 1, 0, 'C')
    pdf.cell(col_widths[2], 10, f"{initial_metrics['Davies-Bouldin Index']:.2f}", 1, 0, 'C')
    pdf.cell(col_widths[3], 10, f"{initial_metrics['Calinski-Harabasz Score']:.2f}", 1, 1, 'C')

    if retrain_metrics:
        pdf.cell(col_widths[0], 10, "Re-Treino", 1, 0, 'C')
        pdf.cell(col_widths[1], 10, f"{retrain_metrics['Silhouette Score']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 10, f"{retrain_metrics['Davies-Bouldin Index']:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[3], 10, f"{retrain_metrics['Calinski-Harabasz Score']:.2f}", 1, 1, 'C')

    pdf.ln(10)

    # Adicionar interpretações
    def add_interpretation(metrics, treino_name):
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 10, txt=f"{treino_name}:", ln=True)
        pdf.multi_cell(0, 10, txt=f"  Silhouette Score: {metrics['Silhouette Score']:.2f} - "
                                  f"{'Excelente' if metrics['Silhouette Score'] > 0.75 else 'Bom' if metrics['Silhouette Score'] > 0.5 else 'Moderado' if metrics['Silhouette Score'] > 0.25 else 'Fraco'} separação entre clusters.")
        pdf.multi_cell(0, 10, txt=f"  Davies-Bouldin Index: {metrics['Davies-Bouldin Index']:.2f} - "
                                  f"{'Muito bom' if metrics['Davies-Bouldin Index'] < 0.5 else 'Bom' if metrics['Davies-Bouldin Index'] < 1.0 else 'Moderado' if metrics['Davies-Bouldin Index'] < 2.0 else 'Fraco'} compactação e separação.")
        pdf.multi_cell(0, 10, txt=f"  Calinski-Harabasz Score: {metrics['Calinski-Harabasz Score']:.2f} - "
                                  f"{'Excelente' if metrics['Calinski-Harabasz Score'] > 2500 else 'Bom' if metrics['Calinski-Harabasz Score'] > 1500 else 'Moderado' if metrics['Calinski-Harabasz Score'] > 500 else 'Fraco'} densidade e separação.")
        pdf.ln(5)

    add_interpretation(initial_metrics, "Treino Inicial")
    if retrain_metrics:
        add_interpretation(retrain_metrics, "Re-Treino")

    # Adicionar gráficos
    metrics_to_plot = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"]
    graphs = []
    for metric in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        labels = ["Treino Inicial"]
        values = [initial_metrics[metric]]
        if retrain_metrics:
            labels.append("Re-Treino")
            values.append(retrain_metrics[metric])
        plt.bar(labels, values, color=['#a8ddb5', '#005a32'], edgecolor='black')
        plt.title(f"{metric} por Treino")
        plt.ylabel(metric)
        plt.xlabel("Treino")
        graph_path = f"temp_{metric.replace(' ', '_')}.png"
        plt.savefig(graph_path)
        plt.close()
        graphs.append(graph_path)

    # Inserir gráficos no PDF
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, txt="Gráficos das Métricas", ln=True)
    pdf.ln(10)

    x_offset = 10
    y_offset = pdf.get_y()
    for i, graph in enumerate(graphs):
        pdf.image(graph, x=x_offset, y=y_offset, w=90, h=70)
        x_offset += 100
        if (i + 1) % 2 == 0:  # Nova linha a cada dois gráficos
            x_offset = 10
            y_offset += 75
        os.remove(graph)  # Remover o arquivo temporário após usá-lo

    # Salvar o PDF no buffer
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# Página final para clustering
def clustering_final_page():
    st.title("Relatório Final do Clustering")

    # Verificar se os dados estão disponíveis
    if "selected_model_name" not in st.session_state or "initial_metrics" not in st.session_state:
        st.error("Nenhuma informação de clustering disponível. Por favor, execute o treino primeiro.")
        return

    # Mostrar o modelo selecionado
    st.subheader("Modelo Selecionado")
    st.write(f"**Modelo:** {st.session_state.selected_model_name}")

    # Exibir métricas do treino inicial
    st.subheader("Métricas do Treino Inicial")
    st.table(fix_dataframe_types(pd.DataFrame([st.session_state.initial_metrics])))

    # Interpretação personalizada para o treino inicial
    initial_metrics = st.session_state.initial_metrics
    st.write("**Interpretação do Treino Inicial:**")
    st.markdown(f"""
    - **Silhouette Score:** {initial_metrics["Silhouette Score"]:.2f} - {"Excelente" if initial_metrics["Silhouette Score"] > 0.75 else "Bom" if initial_metrics["Silhouette Score"] > 0.5 else "Moderado" if initial_metrics["Silhouette Score"] > 0.25 else "Fraco"} separação entre clusters.
    - **Davies-Bouldin Index:** {initial_metrics["Davies-Bouldin Index"]:.2f} - {"Muito bom" if initial_metrics["Davies-Bouldin Index"] < 0.5 else "Bom" if initial_metrics["Davies-Bouldin Index"] < 1.0 else "Moderado" if initial_metrics["Davies-Bouldin Index"] < 2.0 else "Fraco"} compactação e separação.
    - **Calinski-Harabasz Score:** {initial_metrics["Calinski-Harabasz Score"]:.2f} - {"Excelente" if initial_metrics["Calinski-Harabasz Score"] > 2500 else "Bom" if initial_metrics["Calinski-Harabasz Score"] > 1500 else "Moderado" if initial_metrics["Calinski-Harabasz Score"] > 500 else "Fraco"} densidade e separação.
    """)

    # Exibir métricas do re-treino (se disponíveis)
    retrain_silhouette_score = None  # Inicializa como None para evitar erros
    if "retrain_metrics" in st.session_state:
        st.subheader("Métricas do Re-Treino")
        st.table(fix_dataframe_types(pd.DataFrame([st.session_state.retrain_metrics])))

        # Interpretação personalizada para o re-treino
        retrain_metrics = st.session_state.retrain_metrics
        retrain_silhouette_score = retrain_metrics["Silhouette Score"]
        st.write("**Interpretação do Re-Treino:**")
        st.markdown(f"""
        - **Silhouette Score:** {retrain_metrics["Silhouette Score"]:.2f} - {"Excelente" if retrain_metrics["Silhouette Score"] > 0.75 else "Bom" if retrain_metrics["Silhouette Score"] > 0.5 else "Moderado" if retrain_metrics["Silhouette Score"] > 0.25 else "Fraco"} separação entre clusters.
        - **Davies-Bouldin Index:** {retrain_metrics["Davies-Bouldin Index"]:.2f} - {"Muito bom" if retrain_metrics["Davies-Bouldin Index"] < 0.5 else "Bom" if retrain_metrics["Davies-Bouldin Index"] < 1.0 else "Moderado" if retrain_metrics["Davies-Bouldin Index"] < 2.0 else "Fraco"} compactação e separação.
        - **Calinski-Harabasz Score:** {retrain_metrics["Calinski-Harabasz Score"]:.2f} - {"Excelente" if retrain_metrics["Calinski-Harabasz Score"] > 2500 else "Bom" if retrain_metrics["Calinski-Harabasz Score"] > 1500 else "Moderado" if retrain_metrics["Calinski-Harabasz Score"] > 500 else "Fraco"} densidade e separação.
        """)

    # Determinar o melhor modelo considerando apenas o Silhouette Score
    if retrain_silhouette_score is not None and retrain_silhouette_score > initial_metrics["Silhouette Score"]:
        melhor_modelo = "Re-Treino"
        best_metrics = st.session_state.retrain_metrics
        best_model = st.session_state.models[st.session_state.selected_model_name]
    else:
        melhor_modelo = "Treino Inicial"
        best_metrics = st.session_state.initial_metrics
        best_model = st.session_state.models[st.session_state.selected_model_name]

    st.subheader("Melhor Modelo")
    st.success(f"🎉 **{melhor_modelo}** com Silhouette Score: {max(initial_metrics['Silhouette Score'], retrain_silhouette_score or 0):.4f}")


    # **Gráficos Interativos das Métricas**
    st.subheader("Gráfico Interativo de Métricas")
    metrics_to_plot = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"]
    selected_metric = st.selectbox("Selecione a métrica para visualizar:", metrics_to_plot)
    
    # Criar o gráfico
    if selected_metric:
        # Verificar se os dados do re-treino estão presentes
        if "retrain_metrics" in st.session_state:
            # Se o re-treino foi realizado, exibe "Treino Inicial" e "Re-Treino"
            data_to_plot = pd.DataFrame({
                "Treino": ["Treino Inicial", "Re-Treino"],
                selected_metric: [
                    initial_metrics[selected_metric],
                    retrain_metrics[selected_metric]
                ]
            })
        else:
            # Se o re-treino não foi realizado, exibe todas as métricas para "Treino Inicial"
            data_to_plot = pd.DataFrame({
                "Treino": ["Treino Inicial"] * len(metrics_to_plot),
                selected_metric: [initial_metrics[metric] for metric in metrics_to_plot]
            })
    
        # Criar gráfico com base nos dados disponíveis
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(data_to_plot["Treino"], data_to_plot[selected_metric], color=['#a8ddb5', '#005a32'], edgecolor='black')
        ax.set_title(f"Comparação de {selected_metric}", fontsize=14, fontweight='bold')
        ax.set_ylabel(selected_metric, fontsize=12)
        ax.set_xlabel("Treino", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        st.pyplot(fig) 


    # Gerar o relatório PDF
    pdf_buffer = gerar_relatorio_clustering_pdf(
        initial_metrics,
        st.session_state.get("retrain_metrics"),
        melhor_modelo,
        st.session_state
    )

    # Botão para download do relatório em PDF
    pdf_filename = f"Relatorio__{st.session_state.selected_model_name}_{st.session_state.model_type}_{melhor_modelo.replace(' ', '_').lower()}.pdf"
    st.download_button(
        label="Baixar Relatório em PDF",
        data=pdf_buffer,
        file_name=pdf_filename,
        mime="application/pdf"
    )

    # Botão para download do melhor modelo treinado
    model_buffer = BytesIO()
    pickle.dump(best_model, model_buffer)
    model_buffer.seek(0)

    st.download_button(
        label="Baixar Melhor Modelo Treinado",
        data=model_buffer,
        file_name=f"melhor_modelo_{melhor_modelo.replace(' ', '_').lower()}.pkl",
        mime="application/octet-stream"
    )

    # Botão para concluir o processo
    if st.button("Concluir"):
        st.info("Clustering finalizado. Redirecionando para o início...")
        st.session_state.clear()
        st.session_state.step = 'file_upload'
        st.rerun()

def initialize_session_state():
    # Inicializando as variáveis principais de estado
    default_values = {
        'step': 'file_upload',
        'page': 'file_upload',
        'data': None,
        'target_column': None,
        'target_column_confirmed': False,
        'validation_method': None,
        'validation_confirmed': False,
        'model_type': None,
        'model_type_confirmed': False,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'knn_neighbors': 5,
        'rf_estimators': 100,
        'svc_kernel': 'rbf',
        'kmeans_clusters': 3,
        'feature_selection_done': False,
        'X_train_selected': None,
        'X_test_selected': None,
        'model_name': None,
        'selected_model': None,
        'selected_model_name': None,
        'models': {
            "Support Vector Classification (SVC)": SVC(),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KMeans": KMeans(),
            "Clustering Hierárquico": AgglomerativeClustering(),
            "Regressão Linear Simples (RLS)": LinearRegression(),
            "Regressão por Vetores de Suporte (SVR)": SVR(),
        },
        'model_trained': False,
        'clustering_final_page': False,  # Página do relatório final de clustering
        'grid_search_confirmed': False,  # Adicionando grid_search_confirmed ao estado
        'manual_params': {}, # Inicializando manual_params como um dicionário vazio
        'best_params_str': {},
        'treinos_realizados': [],  # Inicializando a lista de treinos realizados
        'scoring_confirmed': False,  # Inicializando scoring_confirmed
        'target_column_type': None,  # Adiciona tipo da coluna alvo
        'selected_scoring': 'F1-Score'  # Inicializando 'selected_scoring' com o valor 'f1'
    }

    # Usando valores padrão para inicializar session_state
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Adicionalmente, você pode garantir que os parâmetros do modelo sejam válidos:
    if st.session_state.knn_neighbors < 1:
        st.session_state.knn_neighbors = 5  # Default para KNN
    if st.session_state.kmeans_clusters < 1:
        st.session_state.kmeans_clusters = 3  # Default para KMeans

# Função principal
def main():
    # Inicialização das variáveis de estado da sessão
    initialize_session_state()

    # Verificar o estado atual da etapa
    #st.write(f"Estado inicial: {st.session_state.step}")

    # Roteamento baseado no estado atual
    if st.session_state.step == 'file_upload':
        upload_file()
    elif st.session_state.step == 'data_preview':
        data_preview()
    elif st.session_state.step == 'missing_values':
        handle_missing_values()
    elif st.session_state.step == 'outlier_detection':
        outlier_detection()
    elif 'outlier_treatment_state' not in st.session_state:
        st.session_state.outlier_treatment_state = {}  # Inicializa como dicionário vazio
    elif st.session_state.step == 'data_summary':
        data_summary()
    elif st.session_state.step == 'model_selection':
        model_selection()  # Chamando a seleção de modelos
    elif st.session_state.step == 'feature_selection':  # Etapa de seleção de features
        feature_selection()
    elif 'scoring_confirmed' not in st.session_state:
        st.session_state.scoring_confirmed = False
    elif st.session_state.step == 'train_and_store_metrics':  # Nova etapa adicionada
        evaluate_and_compare_models()  # Chama a função para comparar os modelos
    elif st.session_state.step == 'clustering_final_page':  # Página final do clustering
        clustering_final_page()
    elif st.session_state.step == 'final_page':  # Página final
        final_page()
    else:
        st.error("Etapa desconhecida. Reiniciando a aplicação.")
        st.session_state.step = 'file_upload'
        st.rerun()

    # Exibir o estado após a execução para depuração
    #st.write(f"Estado final: {st.session_state.step}")


if __name__ == "__main__":
    main()
