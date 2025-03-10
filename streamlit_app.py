########### Bibliotecas Necessárias ###########

# -------------------------------------
# 📌 Bibliotecas para Interface com Utilizador (Streamlit)
# -------------------------------------
import streamlit as st  # Framework para criação de interfaces web interativas
import streamlit.components.v1 as components  # Permite adicionar componentes HTML/CSS personalizados

# -------------------------------------
# 📌 Manipulação e Análise de Dados
# -------------------------------------
import pandas as pd  # Manipulação de DataFrames e séries temporais
import numpy as np  # Operações numéricas e matrizes eficientes

# -------------------------------------
# 📌 Visualização de Dados
# -------------------------------------
import matplotlib.pyplot as plt  # Criação de gráficos estáticos
import seaborn as sns  # Gráficos estatísticos avançados baseados no Matplotlib
import plotly.express as px  # Gráficos interativos e visualizações dinâmicas

# -------------------------------------
# 📌 Modelos de Machine Learning
# -------------------------------------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Modelos baseados em árvores de decisão
from sklearn.linear_model import LogisticRegression, LinearRegression  # Modelos lineares para classificação e regressão
from sklearn.svm import SVC, SVR  # Modelos de Support Vector Machine (SVM) para classificação e regressão
from sklearn.cluster import KMeans, AgglomerativeClustering  # Algoritmos de clustering
from sklearn.neighbors import KNeighborsClassifier  # Modelo de vizinhos mais próximos (KNN)
from sklearn import svm, tree, neighbors  # Modelos adicionais do sklearn

# -------------------------------------
# 📌 Seleção de Features (Atributos)
# -------------------------------------
from mlxtend.feature_selection import SequentialFeatureSelector  # Seleção sequencial de variáveis para otimizar modelos

# -------------------------------------
# 📌 Métricas de Avaliação
# -------------------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,  # Métricas para classificação
    confusion_matrix, classification_report, roc_auc_score,  # Matriz de confusão e análise ROC
    mean_squared_error, mean_absolute_error, r2_score,  # Métricas para regressão
    silhouette_score, davies_bouldin_score, calinski_harabasz_score  # Métricas para clustering
)

# -------------------------------------
# 📌 Pré-Processamento e Pipeline
# -------------------------------------
from sklearn.model_selection import (
    train_test_split,  # Separação entre dados de treino e teste
    KFold, LeaveOneOut, cross_val_score,  # Validação cruzada para avaliar modelos
    GridSearchCV  # Procura de melhores hiperparâmetros usando Grid Search
)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Normalização e codificação de variáveis categóricas
from sklearn.impute import SimpleImputer  # Tratamento de valores ausentes

# -------------------------------------
# 📌 Utilitários Diversos
# -------------------------------------
import os  # Operações no sistema de arquivos (criação de pastas, leitura de arquivos)
import joblib  # Guardar e carregar  modelos treinados
import pickle  # Serialização e desserialização de objetos Python
import json  # Manipulação de arquivos JSON
import requests  # Requisições HTTP para acesso a APIs externas
import unidecode  # Remoção de acentos e normalização de caracteres especiais

# -------------------------------------
# 📌 Manipulação de Arquivos e Dados Binários
# -------------------------------------
from io import BytesIO  # Manipulação de streams binários para arquivos em memória
import tempfile  # Criação de arquivos e diretórios temporários

# -------------------------------------
# 📌 Manipulação de Datas e Cálculos Matemáticos
# -------------------------------------
from datetime import datetime  # Manipulação de datas e horas
from decimal import Decimal  # Precisão extra em cálculos decimais
from fractions import Fraction  # Trabalha com frações matemáticas exatas
from scipy.sparse import csr_matrix  # Representação eficiente de matrizes esparsas
import scipy  # Biblioteca científica para estatísticas, álgebra linear e otimização
import time  # Medição do tempo de execução de processos

# -------------------------------------
# 📌 Bibliotecas para Geração de Relatórios
# -------------------------------------
from fpdf import FPDF  # Criação de documentos PDF programaticamente
from reportlab.lib.pagesizes import letter  # Definição do tamanho da página nos relatórios
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Estilos para formatação de texto
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image  # Estruturação de documentos PDF
from reportlab.lib import colors  # Definição de cores em relatórios
from reportlab.lib.units import inch  # Unidades de medida para layout de documentos



##############################################
# -------------------------------------
# 📌 Função JavaScript para voltar ao topo da página
# -------------------------------------

# Script JavaScript que permite rolar automaticamente para o topo da página
scroll_to_top_js = """
<script>
    function scrollToTop() {
        window.scrollTo(0, 0);  // Move a página para o topo (coordenadas 0,0)
    }
</script>
"""

# Insere o JavaScript na página com Streamlit
# Definição de height=0 e width=0 para evitar que o código ocupe espaço visível na interface
components.html(scroll_to_top_js, height=0, width=0)  

# -------------------------------------
# 📌 Ajustes de Exibição do Pandas Styler
# -------------------------------------

# Define o número máximo de elementos a serem renderizados no Styler do Pandas
pd.set_option("styler.render.max_elements", 2000000)  # Ajustar se necessário para grandes DataFrames

# Configura a exibição de todas as linhas e colunas de um DataFrame
pd.set_option("display.max_rows", None)  # Permite visualizar todas as linhas sem truncamento
pd.set_option("display.max_columns", None)  # Permite visualizar todas as colunas sem truncamento


##############################################
def fix_dataframe_types(df):
    """Corrigir tipos de dados num DataFrame para compatibilidade com PyArrow"""

    # Verificar se o objeto é um Styler e extrair o DataFrame
    if hasattr(df, 'data'):  # Objetos Styler possuem um atributo .data
        df = df.data
    elif hasattr(df, 'render') and not hasattr(df, 'copy'):  # Outra forma de identificar um Styler
        # Para versões mais recentes do pandas
        if hasattr(df, '_data'):
            df = df._data
        # Para versões ainda mais recentes do pandas, onde a estrutura pode ser diferente
        elif hasattr(df, 'data'):
            df = df.data
        # Se ainda não for possível extrair o DataFrame
        else:
            # Tentar converter primeiro para dicionário e depois para DataFrame
            try:
                df = pd.DataFrame(df.to_dict())
            except:
                # Se falhar, retornar um DataFrame vazio
                return pd.DataFrame()
    
    # Se o objeto final não for um DataFrame, retornar um DataFrame vazio
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
        
    # Criar uma cópia do DataFrame para evitar modificar o original
    df_fixed = df.copy()
    
    # Percorrer todas as colunas para corrigir tipos de dados problemáticos
    for col in df_fixed.columns:
        # Converter colunas do tipo Int64 para int64 padrão (evita problemas de compatibilidade)
        if hasattr(df_fixed[col], 'dtype') and str(df_fixed[col].dtype) == 'Int64':
            df_fixed[col] = df_fixed[col].fillna(-1).astype('int64')  # Substituir valores nulos por -1 antes da conversão
        
        # Converter colunas do tipo objeto (strings e dados complexos) para string
        elif df_fixed[col].dtype == 'object':
            try:
                # Tentar converter diretamente para string
                df_fixed[col] = df_fixed[col].astype(str)
            except:
                # Se falhar, aplicar uma conversão manual, garantindo que valores None sejam tratados
                df_fixed[col] = df_fixed[col].apply(lambda x: str(x) if x is not None else "")
    
    # Retornar o DataFrame corrigido
    return df_fixed


##############################################
# -------------------------------------
# 📌 Função para Configurar a Barra Lateral
# -------------------------------------

def configure_sidebar():
    """Configura a barra lateral com o logótipo da instituição e informações sobre a plataforma."""
    
    with st.sidebar:  # Define que os elementos serão adicionados na barra lateral
        st.image(
            "https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg",  # URL da imagem
            width=80,  # Define o tamanho da imagem (largura em pixels)
            caption="Logótipo da Escola"  # Texto exibido abaixo da imagem
        )
        
        # Exibe o nome da plataforma em formato HTML para maior personalização
        st.markdown("<p>MLCase - Plataforma de Machine Learning</p>", unsafe_allow_html=True)
        
        # Exibe o nome da autora com destaque em negrito usando HTML
        st.markdown("<p><b>Autora:</b> Bruna Sousa</p>", unsafe_allow_html=True)

# Chamada da função para configurar a barra lateral
configure_sidebar()


##############################################
import matplotlib
matplotlib.use('Agg')  # Usar backend não interativo
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
##############################################

# -------------------------------------
# 📌 FUNÇÃO DE UPLOAD DE FICHEIROS
# -------------------------------------

# Função para inicializar variáveis de estado na aplicação
def initialize_state():
    """Inicializa variáveis de estado utilizadas na aplicação para gerir diferentes etapas do processo."""
    st.session_state.step = 'data_preview'  # Define o estado inicial como pré-visualização dos dados
    st.session_state.selected_columns = []  # Lista para armazenar colunas selecionadas pelo utilizador
    st.session_state.numeric_types = {}  # Dicionário para armazenar tipos numéricos das variáveis
    st.session_state.variable_types = {}  # Dicionário para armazenar os tipos das variáveis
    st.session_state.treatment_state = {}  # Dicionário para armazenar o estado do tratamento dos dados
    st.session_state.all_treated = False  # Flag para indicar se todos os dados foram tratados

# -------------------------------------
# 📌 Função auxiliar para escolher o delimitador de ficheiros CSV
# -------------------------------------

def choose_delimiter():
    """Permite ao utilizador escolher um delimitador para ficheiros CSV carregados."""
    
    # Lista de delimitadores comuns, incluindo a opção personalizada
    delimiters = [",", ";", "\t", "|", "Outro"]
    
    # Cria um seletor na barra lateral para escolha do delimitador
    delimiter = st.sidebar.selectbox("Escolha o delimitador para CSV", delimiters, index=0)
    
    # Se o utilizador escolher a opção "Outro", permite inserir um delimitador personalizado
    if delimiter == "Outro":
        delimiter = st.sidebar.text_input("Digite o delimitador personalizado:")
    
    return delimiter

# -------------------------------------
# 📌 Função para a etapa de upload do ficheiro
# -------------------------------------

def upload_file():
    """Permite ao utilizador carregar um ficheiro de dados para a plataforma."""
    
    st.title("MLCase - Plataforma de Machine Learning")  # Título principal da aplicação

    # Seleção do tipo de ficheiro a ser carregado
    file_type = st.sidebar.selectbox("Selecione o tipo de arquivo", ["CSV", "Excel", "JSON"])
    delimiter = ","  # Define o delimitador padrão para CSV

    # Processo de upload conforme o tipo de ficheiro selecionado
    if file_type == "CSV":
        delimiter = choose_delimiter()  # Permite selecionar um delimitador para o CSV
        file = st.sidebar.file_uploader("Carregar arquivo", type=["csv"])  # Botão de upload
    elif file_type == "Excel":
        file = st.sidebar.file_uploader("Carregar arquivo", type=["xlsx", "xls"])  # Upload de ficheiro Excel
    elif file_type == "JSON":
        file = st.sidebar.file_uploader("Carregar arquivo", type=["json"])  # Upload de ficheiro JSON

    # Se um ficheiro for carregado, tenta processá-lo
    if file is not None:
        try:
            # Chama a função de carregamento de dados e inicializa as variáveis de estado
            st.session_state.data = load_data(file_type, file, delimiter)
            initialize_state()
            st.sidebar.success(f"Conjunto de dados {file_type} carregado com sucesso!")  # Mensagem de sucesso

            # Botão para avançar para a próxima etapa (pré-visualização dos dados)
            if st.sidebar.button("Dados Carregados"):
                st.session_state.step = 'data_preview'  # Atualiza o estado para a pré-visualização
                st.stop()  # Para a execução para refletir as mudanças

        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {e}")  # Exibe mensagem de erro caso algo corra mal

# -------------------------------------
# 📌 Função para carregar dados com cache (evita recarregamento desnecessário)
# -------------------------------------

@st.cache_data  # Usa cache para evitar recarregar os dados várias vezes
def load_data(file_type, file, delimiter):
    """Carrega um ficheiro de dados conforme o tipo selecionado pelo utilizador."""
    
    if file_type == "CSV":
        return pd.read_csv(file, delimiter=delimiter)  # Carrega dados CSV com o delimitador escolhido
    elif file_type == "Excel":
        return pd.read_excel(file)  # Carrega ficheiro Excel
    elif file_type == "JSON":
        return pd.read_json(file)  # Carrega ficheiro JSON

##############################################
# -------------------------------------
# 📌 FUNÇÃO DE SELEÇÃO DE COLUNAS
# -------------------------------------

# Função para pré-visualizar os dados e permitir a seleção de colunas e tipos de variáveis
def data_preview():
    """Permite visualizar os dados carregados, selecionar colunas e definir os seus tipos."""

    # Exibir uma pré-visualização dos primeiros registos do dataset (com correção de tipos)
    st.subheader("Pré-visualização dos dados")
    st.dataframe(fix_dataframe_types(st.session_state.data.head()))  # Corrige os tipos antes da exibição

    # Obter a lista de colunas do dataset
    columns = st.session_state.data.columns.tolist()

    # Criar uma caixa de seleção múltipla para escolher quais colunas utilizar
    selected_columns = st.multiselect("Colunas", columns, columns)  # Por defeito, todas as colunas são selecionadas
    st.session_state.selected_columns = selected_columns  # Guardar as colunas selecionadas no estado global

    # Preservar transformações no estado global
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = st.session_state.data.copy()  # Criar uma cópia inicial dos dados
    else:
        # Atualizar os dados filtrados apenas com as colunas selecionadas, mantendo transformações já aplicadas
        st.session_state.filtered_data = st.session_state.data[selected_columns]

    # Se houver colunas selecionadas, permitir a identificação dos tipos de variáveis
    if selected_columns:
        st.subheader("Identificar tipos de variáveis")

        # Inicializar dicionário para armazenar os tipos de variáveis, caso ainda não exista
        if 'variable_types' not in st.session_state:
            st.session_state.variable_types = {}

        variable_types = st.session_state.variable_types
        st.session_state.numeric_types = {}  # Dicionário para armazenar os tipos numéricos

        # Percorrer cada coluna selecionada para definir os tipos de variáveis
        for col in selected_columns:
            # Criar um seletor para definir se a variável é Numérica, Categórica ou Data
            var_type = st.selectbox(
                f"Tipo de variável para {col}",
                ["Numérica", "Categórica", "Data"],
                index=0 if pd.api.types.is_numeric_dtype(st.session_state.filtered_data[col]) else 1,
                key=f"var_{col}"  # Cada seletor tem uma chave única para evitar conflitos
            )
            variable_types[col] = var_type  # Guardar o tipo selecionado

            # Se a variável for numérica, permitir configurar o tipo específico
            if var_type == "Numérica":
                num_type = st.selectbox(
                    f"Tipo numérico para {col}",
                    ["Int", "Float", "Complex", "Dec", "Frac", "Bool"],
                    index=0 if pd.api.types.is_integer_dtype(st.session_state.filtered_data[col]) else 1,
                    key=f"num_{col}"  # Chave única para o seletor de tipo numérico
                )
                st.session_state.numeric_types[col] = num_type  # Guardar o tipo numérico no estado global

                # Discretização da variável (conversão para categorias)
                # Verifica primeiro se a coluna já foi discretizada
                if col not in st.session_state.filtered_data.columns or pd.api.types.is_numeric_dtype(st.session_state.filtered_data[col]):
                    if st.checkbox(f"Discretizar {col}?", key=f"discretize_{col}"):
                        discretize_column(col)  # Aplica a função de discretização
                else:
                    st.write(f"Coluna {col} já foi discretizada.")  # Informação para o utilizador

        # Atualizar o estado global com os tipos de variáveis definidos
        st.session_state.variable_types = variable_types

    # Criar uma cópia dos dados filtrados para manter alterações recentes
    st.session_state.filtered_data = st.session_state.filtered_data.copy()

    # -------------------------------------
    # 📌 Navegação entre etapas
    # -------------------------------------

    col1, col2 = st.columns(2)  # Criar duas colunas para os botões "Voltar" e "Próxima etapa"

    # Botão para voltar à etapa anterior
    with col1:
        if st.button("Voltar"):
            # Apagar estados salvos explicitamente para evitar conflitos
            keys_to_reset = [
                'filtered_data', 'selected_columns', 'variable_types',
                'numeric_types', 'treatment_state'
            ]
            for key in keys_to_reset:
                st.session_state.pop(key, None)  # Remove do estado se existir

            # Restaurar os dados originais
            st.session_state.data = st.session_state.data.copy()

            # Voltar para a etapa de upload do ficheiro
            st.session_state.step = 'file_upload'
            st.rerun()  # Recarregar a aplicação para refletir as mudanças

    # Botão para avançar para a próxima etapa
    with col2:
        if st.button("Próxima etapa"):
            apply_numeric_types()  # Aplicar os tipos numéricos definidos pelo utilizador
            st.session_state.step = 'missing_values'  # Atualizar o estado para a etapa seguinte
            st.rerun()  # Recarregar a aplicação para refletir as alterações


# -------------------------------------
# 📌 Função para Aplicar Tipos Numéricos às Colunas Filtradas
# -------------------------------------

def apply_numeric_types():
    """Aplica os tipos numéricos definidos pelo utilizador às colunas filtradas no dataset."""
    
    # Percorre todas as colunas que têm tipos numéricos definidos pelo utilizador
    for col, num_type in st.session_state.numeric_types.items():
        # Verifica se a coluna ainda existe no conjunto de dados filtrado
        if col in st.session_state.filtered_data.columns:
            # Converte a coluna para o tipo numérico selecionado
            st.session_state.filtered_data[col] = convert_numeric_type(st.session_state.filtered_data[col], num_type)

# -------------------------------------
# 📌 Função para Conversão de Tipos de Dados Numéricos
# -------------------------------------

def convert_numeric_type(series, num_type):
    """
    Converte uma série de dados para o tipo numérico especificado.
    
    Parâmetros:
    - series: pd.Series -> Coluna do DataFrame a ser convertida.
    - num_type: str -> Tipo numérico desejado ("Int", "Float", "Complex", "Dec", "Frac", "Bool", "Date", "Duration").

    Retorna:
    - pd.Series convertida para o tipo especificado ou a mesma série original caso ocorra um erro.
    """
    
    try:
        # Conversão para número inteiro (Int64)
        if num_type == "Int":
            return pd.to_numeric(series, errors='coerce').astype('Int64')  # Mantém valores nulos compatíveis com Pandas

        # Conversão para número decimal (Float)
        elif num_type == "Float":
            return pd.to_numeric(series, errors='coerce').astype(float)

        # Conversão para número complexo
        elif num_type == "Complex":
            return pd.to_numeric(series, errors='coerce').apply(lambda x: complex(x) if pd.notnull(x) else np.nan)

        # Conversão para Decimal (melhor precisão para cálculos financeiros)
        elif num_type == "Dec":
            return series.apply(lambda x: Decimal(x) if pd.notnull(x) else np.nan)

        # Conversão para Fração (representação matemática exata)
        elif num_type == "Frac":
            return series.apply(lambda x: Fraction(x) if pd.notnull(x) else np.nan)

        # Conversão para Booleano (True/False)
        elif num_type == "Bool":
            return series.apply(lambda x: str(x).strip().lower() in ['true', '1'])

        # Conversão para Data/Hora
        elif num_type == "Date":
            return pd.to_datetime(series, errors='coerce')

        # Conversão para Duração/Intervalo de Tempo
        elif num_type == "Duration":
            return pd.to_timedelta(series, errors='coerce')

        # Se o tipo especificado não estiver listado, retorna a série original sem alterações
        else:
            return series

    except Exception as e:
        # Exibe um erro no Streamlit caso ocorra um problema na conversão
        st.error(f"Erro ao converter coluna {series.name} para tipo {num_type}: {e}")


# -------------------------------------
# 📌 Função para Discretizar uma Coluna Numérica
# -------------------------------------

def discretize_column(col):
    """Permite ao utilizador discretizar uma coluna numérica, transformando-a em categorias definidas manualmente."""

    # -------------------------------------
    # 📌 Seção de Ajuda - Explicação sobre Discretização
    # -------------------------------------
    
    # Explicação interativa sobre como definir bins (intervalos) e labels (categorias)
    with st.expander("Como preencher os bins e labels?"):
        st.write("**Bins:** Intervalos numéricos para discretização.")
        st.write("**Labels:** Nomeiam os intervalos.")
        st.write("**Exemplo:**")
        st.write("- **Bins:** -2,1,2,6,inf")
        st.write("- **Labels:** Baixo, Médio, Alto, Muito Alto")

    # -------------------------------------
    # 📌 Diagnóstico Inicial Antes da Discretização
    # -------------------------------------

    st.write("### Diagnóstico antes da discretização:")
    st.write(f"- **Mínimo:** {st.session_state.filtered_data[col].min()}")  # Valor mínimo da coluna
    st.write(f"- **Máximo:** {st.session_state.filtered_data[col].max()}")  # Valor máximo da coluna
    st.write(f"- **Média:** {st.session_state.filtered_data[col].mean():.2f}")  # Média da coluna
    st.write(f"- **Mediana:** {st.session_state.filtered_data[col].median():.2f}")  # Mediana da coluna
    st.write(f"- **Valores ausentes antes:** {st.session_state.filtered_data[col].isna().sum()}")  # Contagem de valores nulos

    # -------------------------------------
    # 📌 Entrada de Dados do Utilizador (Bins e Labels)
    # -------------------------------------

    # Caixa de texto para o utilizador inserir os bins (intervalos numéricos)
    bins_input = st.text_input(
        f"Digite os bins para {col} (separados por vírgulas)",
        value="-2,1,2,6,inf", key=f"bins_{col}"
    )

    # Caixa de texto para o utilizador inserir os labels (nomes das categorias correspondentes aos bins)
    labels_input = st.text_input(
        f"Digite os labels para {col} (separados por vírgulas)",
        value="Baixo,Médio,Alto,Muito Alto", key=f"labels_{col}"
    )

    # -------------------------------------
    # 📌 Aplicação da Discretização Após Confirmação
    # -------------------------------------

    # Se o utilizador clicar no botão, iniciar a conversão
    if st.button(f"Confirmar Discretização para {col}", key=f"confirm_{col}"):

        # Verificar se o utilizador preencheu os bins e labels corretamente
        if bins_input and labels_input:
            try:
                # Converter a string de bins para uma lista de valores numéricos (float)
                bins = list(map(float, bins_input.split(',')))

                # Converter a string de labels para uma lista de nomes de categorias
                labels = labels_input.split(',')

                # -------------------------------------
                # 📌 Validação de Dados Antes da Conversão
                # -------------------------------------

                # O número de labels deve ser igual ao número de bins menos um
                if len(labels) != len(bins) - 1:
                    st.error(f"O número de labels deve ser igual ao número de bins menos um para a coluna {col}.")

                else:
                    # Converter a coluna para tipo numérico para evitar erros
                    st.session_state.filtered_data[col] = pd.to_numeric(
                        st.session_state.filtered_data[col], errors='coerce'
                    )

                    # Preencher valores ausentes com a mediana da coluna
                    median_value = st.session_state.filtered_data[col].median()
                    st.session_state.filtered_data[col].fillna(median_value, inplace=True)

                    # Diagnóstico após preenchimento de valores ausentes
                    st.write(f"Valores ausentes após preenchimento: {st.session_state.filtered_data[col].isna().sum()}")

                    # -------------------------------------
                    # 📌 Aplicação da Discretização
                    # -------------------------------------

                    # Criar categorias com base nos bins e labels definidos pelo utilizador
                    categorized = pd.cut(
                        st.session_state.filtered_data[col],  # Coluna de dados a ser discretizada
                        bins=bins,  # Intervalos definidos
                        labels=labels,  # Nomes das categorias correspondentes
                        include_lowest=True  # Inclui o menor valor nos intervalos
                    )

                    # Converter para tipo categórico
                    categorized = categorized.astype('category')

                    # Adicionar uma categoria extra para valores fora do intervalo definido
                    categorized = categorized.cat.add_categories(["Fora do Intervalo"])
                    categorized = categorized.fillna("Fora do Intervalo")  # Substituir valores não categorizados

                    # -------------------------------------
                    # 📌 Atualização do Estado Global e Diagnóstico Final
                    # -------------------------------------

                    # Salvar a coluna discretizada no dataset filtrado
                    st.session_state.filtered_data[col] = categorized

                    # Criar uma nova cópia do dataset para garantir a consistência dos dados
                    st.session_state.filtered_data = st.session_state.filtered_data.copy()

                    # Mensagem de sucesso
                    st.success(f"Coluna {col} discretizada com sucesso!")

                    # Exibir o tipo de dados final da coluna
                    st.write(st.session_state.filtered_data[col].dtype)

                    # Exibir as categorias únicas geradas
                    st.write(st.session_state.filtered_data[col].unique())

                    # Exibir uma pré-visualização dos dados após a discretização
                    st.write("Pré-visualização dos dados após discretização:")
                    st.dataframe(fix_dataframe_types(st.session_state.filtered_data.head()))

            except ValueError as e:
                # Mensagem de erro caso a conversão falhe
                st.error(f"Erro ao discretizar {col}: {e}")



##############################################
# -------------------------------------
# 📌 FUNÇÃO DE TRATAMENTO DE VALORES OMISSOS (MISSING VALUES)
# -------------------------------------

# -------------------------------------
# 📌 Função para destacar valores ausentes no DataFrame
# -------------------------------------

def highlight_missing():
    """Aplica um estilo ao DataFrame, destacando células com valores ausentes em amarelo."""

    # Função interna que aplica a cor amarela às células com valores nulos (NaN)
    def highlight_na(s):
        return ['background-color: yellow' if pd.isnull(v) else '' for v in s]

    # Aplica o estilo ao DataFrame filtrado e retorna o objeto Styler
    return st.session_state.filtered_data.style.apply(highlight_na, subset=st.session_state.filtered_data.columns)

# -------------------------------------
# 📌 Função para formatar valores na tabela
# -------------------------------------

def format_table():
    """Formata os valores do DataFrame para exibição, ajustando casas decimais e representações de NaN."""
    
    # Criar uma cópia do DataFrame para evitar modificar os dados originais
    formatted_df = st.session_state.filtered_data.copy()

    # Iterar sobre todas as colunas do DataFrame
    for col in formatted_df.columns:
        # Verificar se a coluna contém valores numéricos
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            # Formatar os valores numéricos para exibição com 2 casas decimais
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else 'NaN')

    return formatted_df  # Retorna o DataFrame formatado

# -------------------------------------
# 📌 Função para exibir a pré-visualização dos dados com tipos de variáveis
# -------------------------------------

def show_preview_with_types(variable_types):
    """Exibe os dados com uma pré-visualização dos tipos de variáveis identificados."""

    # Título da seção
    st.subheader("Pré-visualização dos dados com tipos de variáveis")

    # Exibir os tipos de variáveis definidos pelo utilizador
    st.write("Tipos de variáveis:")
    st.write(variable_types)

    # Formatar os dados antes da exibição
    formatted_df = format_table()

    # Aplicar destaque para valores ausentes e corrigir tipos de dados antes de exibir
    st.dataframe(fix_dataframe_types(highlight_missing(formatted_df)))


# -------------------------------------
# 📌 Função para Aplicar Tratamento de Valores Ausentes
# -------------------------------------

def apply_missing_value_treatment(column, method, constant_value=None):
    """Aplica um tratamento específico para valores ausentes numa coluna selecionada do dataset."""

    # Usa diretamente os dados filtrados armazenados no estado global
    data = st.session_state.filtered_data

    # Verifica se a coluna é numérica
    if pd.api.types.is_numeric_dtype(data[column]):
        # Substituir valores ausentes pela média da coluna
        if method == "Média":
            data[column].fillna(data[column].mean(), inplace=True)

        # Substituir valores ausentes pela mediana da coluna
        elif method == "Mediana":
            data[column].fillna(data[column].median(), inplace=True)

        # Substituir valores ausentes pela moda (valor mais frequente) da coluna
        elif method == "Moda":
            data[column].fillna(data[column].mode().iloc[0], inplace=True)

        # Excluir linhas onde há valores ausentes nesta coluna
        elif method == "Excluir":
            data.dropna(subset=[column], inplace=True)

        # Substituir por um valor constante definido pelo utilizador
        elif method == "Valor constante" and constant_value is not None:
            data[column].fillna(constant_value, inplace=True)

    # Se a coluna for categórica (texto, categorias, etc.)
    else:
        # Substituir valores ausentes pela moda (valor mais frequente)
        if method == "Substituir por moda":
            data[column].fillna(data[column].mode().iloc[0], inplace=True)

        # Substituir valores ausentes por um valor fixo definido pelo utilizador
        elif method == "Substituir por valor constante" and constant_value is not None:
            data[column].fillna(constant_value, inplace=True)

        # Não faz nada (mantém os valores ausentes)
        elif method == "Manter valores ausentes":
            pass  

        # Excluir linhas com valores ausentes nesta coluna
        elif method == "Excluir":
            data.dropna(subset=[column], inplace=True)

    # Atualiza os dados processados no estado global
    st.session_state.filtered_data = data

# -------------------------------------
# 📌 Função para Selecionar Automaticamente o Método de Tratamento de Valores Ausentes
# -------------------------------------

def auto_select_method(column_name):
    """Seleciona automaticamente o melhor método para tratar valores ausentes numa coluna."""

    # Obtém a coluna a partir dos dados filtrados
    column = st.session_state.filtered_data[column_name]

    # Calcula a percentagem de valores ausentes na coluna
    missing_percentage = column.isnull().sum() / len(column)

    # Para colunas numéricas
    if pd.api.types.is_numeric_dtype(column):
        if missing_percentage > 0.5:
            return "Excluir"  # Se mais de 50% dos valores estão ausentes, sugere excluir a coluna
        else:
            return "Substituir por Mediana"  # Caso contrário, sugere substituir pela mediana

    # Para colunas categóricas (texto, categorias)
    else:
        if missing_percentage > 0.5:
            return "Excluir"  # Se mais de 50% dos valores estão ausentes, sugere excluir a coluna
        else:
            return "Substituir por Moda"  # Caso contrário, sugere substituir pela moda (valor mais frequente)

# -------------------------------------
# 📌 Função para Exibir Tabela com Valores Ausentes
# -------------------------------------

def display_missing_values(dataframe):
    """Exibe uma tabela com a contagem de valores ausentes em cada coluna do dataset."""

    # Conta o número de valores ausentes por coluna
    missing_data = dataframe.isnull().sum()

    # Mantém apenas as colunas que possuem valores ausentes
    missing_data = missing_data[missing_data > 0]
    
    # Converte para DataFrame para melhor visualização
    missing_data = missing_data.reset_index()
    missing_data.columns = ['Coluna', 'Valores Ausentes']

    # Se houver valores ausentes, exibir a tabela
    if not missing_data.empty:
        st.write("Tabela de valores ausentes:")
        st.dataframe(fix_dataframe_types(missing_data))  # Aplica correções de tipo antes de exibir
    else:
        st.write("Não há valores ausentes.")  # Mensagem caso não existam valores em falta

# -------------------------------------
# 📌 FUNÇÃO PARA MOSTRAR E TRATAR VALORES AUSENTES
# -------------------------------------

def handle_missing_values():
    """Gerencia o tratamento de valores ausentes no dataset carregado."""

    # Exibe o título da seção no Streamlit
    st.subheader("Tratamento de Valores Ausentes")

    # Obtém os dados filtrados armazenados no estado da sessão
    filtered_data = st.session_state.get('filtered_data', None)

    # -------------------------------------
    # 📌 Verificação Inicial dos Dados
    # -------------------------------------

    # Verifica se há dados carregados e não estão vazios
    if filtered_data is not None and not filtered_data.empty:

        # -------------------------------------
        # 📌 Função Interna para Exibir Valores Ausentes
        # -------------------------------------

        def display_missing_values(df):
            """Gera uma tabela resumida com a contagem de valores ausentes por coluna."""

            # Conta a quantidade de valores ausentes em cada coluna
            missing_data = df.isnull().sum()

            # Mantém apenas as colunas que possuem valores ausentes
            missing_data = missing_data[missing_data > 0]

            # Exibe os valores ausentes caso existam
            if not missing_data.empty:
                st.write("Resumo dos Valores Ausentes:")
                st.dataframe(fix_dataframe_types(missing_data.rename("Total de Valores Ausentes")))
            else:
                st.success("Não há valores ausentes nos dados.")  # Exibe uma mensagem caso não haja valores ausentes

        # Exibir o resumo dos valores ausentes no dataset
        display_missing_values(filtered_data)

        # -------------------------------------
        # 📌 Configuração das Opções de Tratamento de Valores Ausentes
        # -------------------------------------

        # Verifica se existem valores ausentes em qualquer coluna
        has_missing_values = filtered_data.isnull().any().any()

        if has_missing_values:
            # Inicializar dicionário de tratamento no estado global, caso ainda não exista
            if 'treatment_state' not in st.session_state:
                st.session_state.treatment_state = {
                    col: {"method": None, "constant": None}
                    for col in filtered_data.columns
                }

            # Percorre cada coluna que possui valores ausentes para exibir opções de tratamento
            for col in filtered_data.columns:
                if filtered_data[col].isnull().sum() > 0:
                    col_state = st.session_state.treatment_state.get(col, {"method": None, "constant": None})
                    is_numeric = pd.api.types.is_numeric_dtype(filtered_data[col])

                    # -------------------------------------
                    # 📌 Tratamento de Valores Ausentes em Colunas Numéricas
                    # -------------------------------------

                    if is_numeric:
                        # Opções disponíveis para tratamento de valores ausentes em variáveis numéricas
                        options = ["Substituir por Média", "Substituir por Mediana", "Substituir por Moda", 
                                   "Substituir por Valor Constante", "Excluir", "Manter Valores Ausentes"]
                        
                        # Seletor para escolher o método de tratamento
                        missing_value_method = st.selectbox(
                            f"Método para tratar valores ausentes em {col}",
                            options,
                            index=options.index(col_state["method"]) if col_state["method"] in options else 0,
                            key=f"missing_value_{col}"
                        )

                        # Definir valor constante caso o utilizador escolha essa opção
                        constant_value = None
                        if missing_value_method == "Substituir por Valor Constante":
                            constant_value = st.text_input(
                                f"Digite o valor constante para {col}:",
                                value=col_state["constant"] if col_state["constant"] else '',
                                key=f"constant_{col}"
                            )

                    # -------------------------------------
                    # 📌 Tratamento de Valores Ausentes em Colunas Categóricas
                    # -------------------------------------

                    else:
                        # Opções disponíveis para colunas categóricas
                        options = ["Substituir por Moda", "Substituir por Valor Constante", "Manter Valores Ausentes", "Excluir"]
                        
                        # Seletor para escolher o método de tratamento
                        missing_value_method = st.selectbox(
                            f"Método para tratar valores ausentes em {col}",
                            options,
                            index=options.index(col_state["method"]) if col_state["method"] in options else 0,
                            key=f"cat_missing_value_{col}"
                        )

                        # Definir valor constante caso o utilizador escolha essa opção
                        constant_value = None
                        if missing_value_method == "Substituir por Valor Constante":
                            constant_value = st.text_input(
                                f"Digite o valor constante para {col}:",
                                value=col_state["constant"] if col_state["constant"] else '',
                                key=f"cat_constant_{col}"
                            )

                    # Atualizar o estado global com as escolhas do utilizador para essa coluna
                    st.session_state.treatment_state[col] = {"method": missing_value_method, "constant": constant_value}

            # -------------------------------------
            # 📌 Aplicação dos Tratamentos Escolhidos
            # -------------------------------------

            if st.button("Aplicar tratamentos"):
                for col, treatment in st.session_state.treatment_state.items():
                    method = treatment["method"]
                    constant_value = treatment["constant"]

                    # Aplicar o método selecionado para tratamento dos valores ausentes
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

                # Atualizar os dados processados no estado global
                st.session_state.data = filtered_data.copy()

                # Mensagem de sucesso
                st.success("Tratamentos aplicados com sucesso!")

        # -------------------------------------
        # 📌 Navegação entre Etapas
        # -------------------------------------

        col1, col2 = st.columns(2)

        # Botão para voltar à etapa anterior
        with col1:
            if st.button("Voltar"):
                st.session_state.step = 'data_preview'
                st.rerun()

        # Botão para avançar para a próxima etapa
        with col2:
            if st.button("Próxima etapa"):
                st.session_state.step = 'outlier_detection'
                st.rerun()

    else:
        # Caso não haja dados disponíveis, exibir uma mensagem de erro
        st.error("Nenhum dado disponível para tratamento de valores ausentes.")


##############################################
# -------------------------------------
# 📌 FUNÇÃO DE TRATAMENTO DE OUTLIERS (VALORES EXTREMOS)
# -------------------------------------

# -------------------------------------
# 📌 Função para Detetar e Calcular Informações sobre Outliers
# -------------------------------------

@st.cache_data  # Usa cache para evitar recálculo desnecessário ao interagir com a aplicação
def calculate_outliers(columns, data):
    """
    Identifica e calcula estatísticas sobre outliers em variáveis numéricas.

    Parâmetros:
    - columns: lista com os nomes das colunas a serem analisadas.
    - data: DataFrame contendo os dados.

    Retorna:
    - variables_with_outliers: Lista com as variáveis que possuem outliers.
    - outlier_summary: Lista de dicionários com informações detalhadas sobre os outliers identificados.
    """

    # Lista para armazenar os nomes das variáveis que contêm outliers
    variables_with_outliers = []

    # Lista para armazenar o resumo estatístico dos outliers encontrados
    outlier_summary = []

    # Percorre todas as colunas selecionadas para análise de outliers
    for col in columns:
        # Verifica se a coluna contém dados numéricos antes de continuar a análise
        if pd.api.types.is_numeric_dtype(data[col]):

            # -------------------------------------
            # 📌 Cálculo do Intervalo Interquartil (IQR)
            # -------------------------------------

            # Primeiro quartil (Q1) - 25% dos dados estão abaixo deste valor
            Q1 = data[col].quantile(0.25)

            # Terceiro quartil (Q3) - 75% dos dados estão abaixo deste valor
            Q3 = data[col].quantile(0.75)

            # Intervalo Interquartil (IQR) - Diferença entre Q3 e Q1
            IQR = Q3 - Q1

            # Definição dos limites para deteção de outliers
            lower_bound = Q1 - 1.5 * IQR  # Limite inferior
            upper_bound = Q3 + 1.5 * IQR  # Limite superior

            # -------------------------------------
            # 📌 Identificação de Outliers
            # -------------------------------------

            # Contagem de outliers, ou seja, valores que estão abaixo do limite inferior ou acima do superior
            num_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])

            # Se forem encontrados outliers na coluna, armazenar os resultados
            if num_outliers > 0:
                # Calcular a percentagem de outliers em relação ao total de dados na variável
                percentage_outliers = (num_outliers / len(data[col])) * 100

                # Adicionar o nome da variável à lista de variáveis com outliers
                variables_with_outliers.append(col)

                # Criar um dicionário com o resumo estatístico dos outliers na variável analisada
                outlier_summary.append({
                    "Variável": col,
                    "Total de Outliers": num_outliers,
                    "Percentagem de Outliers (%)": round(percentage_outliers, 2)
                })

    # Retorna a lista de variáveis que possuem outliers e o resumo estatístico
    return variables_with_outliers, outlier_summary


# Interface de detecção e tratamento de outliers
# -------------------------------------
# 📌 FUNÇÃO DE DETEÇÃO E TRATAMENTO DE OUTLIERS
# -------------------------------------

def outlier_detection():
    """Realiza a deteção e o tratamento de outliers (valores extremos) em variáveis numéricas do dataset."""

    # Exibir o título da seção no Streamlit
    st.subheader("Deteção de Outliers")

    # -------------------------------------
    # 📌 Armazenamento dos Dados Originais
    # -------------------------------------

    # Se for a primeira execução, armazenar uma cópia dos dados originais
    if 'original_data' not in st.session_state:
        st.session_state.original_data = st.session_state.data.copy()

    # -------------------------------------
    # 📌 Boxplot Inicial (Visualização dos Dados Antes do Tratamento)
    # -------------------------------------

    st.write("### Boxplot Inicial (Dados Originais)")
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.original_data.boxplot(ax=ax)  # Criar boxplot para visualizar outliers
    plt.xticks(rotation=45)  # Ajustar rotação dos rótulos do eixo X
    st.pyplot(fig)  # Exibir gráfico no Streamlit

    # -------------------------------------
    # 📌 Inicializar Estados Globais Necessários
    # -------------------------------------

    # Armazena colunas que já passaram por tratamento
    if 'treated_columns' not in st.session_state:
        st.session_state.treated_columns = []

    # Armazena detalhes sobre os outliers identificados
    if 'outlier_details' not in st.session_state:
        st.session_state.outlier_details = {}

    # Armazena os limites iniciais dos outliers (antes do tratamento)
    if 'initial_limits' not in st.session_state:
        st.session_state.initial_limits = {}

    # Lista de colunas que possuem outliers
    if 'columns_with_outliers' not in st.session_state:
        st.session_state.columns_with_outliers = []

    # Estado global para armazenar as decisões do utilizador sobre tratamento de outliers
    if 'outlier_treatment_state' not in st.session_state:
        st.session_state.outlier_treatment_state = {}

    # Flag para indicar se todos os outliers foram tratados
    if 'all_outliers_treated' not in st.session_state:
        st.session_state.all_outliers_treated = False

    # -------------------------------------
    # 📌 Verificação da Disponibilidade dos Dados
    # -------------------------------------

    if 'data' not in st.session_state or st.session_state.data is None:
        st.error("Os dados não estão carregados! Volte para a etapa anterior.")
        return

    # -------------------------------------
    # 📌 Identificação de Outliers
    # -------------------------------------

    # Selecionar apenas as colunas numéricas do dataset
    numeric_columns = list(st.session_state.data.select_dtypes(include=[np.number]).columns)

    # Lista para armazenar resumo dos outliers
    outlier_summary = []

    # Percorrer todas as colunas numéricas para calcular limites e identificar outliers
    for col in numeric_columns:

        # Ignorar colunas que já foram tratadas
        if col in st.session_state.treated_columns:
            continue

        # Calcular o primeiro quartil (Q1) e o terceiro quartil (Q3)
        Q1 = st.session_state.data[col].quantile(0.25)
        Q3 = st.session_state.data[col].quantile(0.75)

        # Calcular o intervalo interquartil (IQR)
        IQR = Q3 - Q1

        # Definir limites inferior e superior para identificação de outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Contar outliers normais (fora do intervalo IQR)
        total_outliers = len(st.session_state.data[(st.session_state.data[col] < lower_bound) | 
                                                   (st.session_state.data[col] > upper_bound)])

        # Contar outliers severos (fora do intervalo 3*IQR)
        total_severe_outliers = len(st.session_state.data[(st.session_state.data[col] < (Q1 - 3.0 * IQR)) | 
                                                           (st.session_state.data[col] > (Q3 + 3.0 * IQR))])

        # Se a variável contiver outliers, armazenar detalhes
        if total_outliers > 0:
            st.session_state.initial_limits[col] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            st.session_state.outlier_details[col] = {
                "total_outliers": total_outliers,
                "total_severe_outliers": total_severe_outliers,
                "skewness": st.session_state.data[col].skew()  # Assimetria da distribuição
            }

            # Adicionar ao resumo estatístico
            outlier_summary.append({
                "Nome variável": col,
                "Total de outliers": total_outliers,
                "Total de outliers severos": total_severe_outliers
            })

            # Adicionar à lista de colunas com outliers
            if col not in st.session_state.columns_with_outliers:
                st.session_state.columns_with_outliers.append(col)

    # Salvar o resumo inicial no estado global
    st.session_state.initial_outlier_summary = outlier_summary

    # -------------------------------------
    # 📌 Verificar se Restam Outliers para Tratar
    # -------------------------------------

    remaining_outliers = [col for col in st.session_state.columns_with_outliers 
                          if col not in st.session_state.treated_columns]

    if not remaining_outliers:
        if not outlier_summary and not st.session_state.columns_with_outliers:
            st.success("Nenhum outlier detetado nas variáveis numéricas!")
        else:
            st.success("Todos os outliers detetados foram tratados!")
    else:
        st.write("Resumo dos Outliers:")
        st.dataframe(fix_dataframe_types(pd.DataFrame(outlier_summary)))

    # -------------------------------------
    # 📌 Exibição e Tratamento de Outliers Restantes
    # -------------------------------------

    for col in remaining_outliers:
        st.write(f"**Diagnóstico para {col}:**")
        details = st.session_state.outlier_details[col]
        st.write(f"- Total de Registos: {len(st.session_state.data)}")
        st.write(f"- Outliers: {details['total_outliers']} ({(details['total_outliers'] / len(st.session_state.data)):.2%})")
        st.write(f"- Outliers Severos: {details['total_severe_outliers']} ({(details['total_severe_outliers'] / len(st.session_state.data)):.2%})")
        st.write(f"- Assimetria (Skewness): {details['skewness']:.2f}")

        # Sugestão automática de método de tratamento
        if col not in st.session_state.outlier_treatment_state:
            suggested_method = auto_select_outlier_treatment(
                col, st.session_state.data, st.session_state.initial_limits[col]["lower_bound"], st.session_state.initial_limits[col]["upper_bound"]
            )
            st.session_state.outlier_treatment_state[col] = suggested_method

        # Seletor de método de tratamento
        method = st.selectbox(
            f"Selecione o método para tratar outliers em {col}",
            ["Sem Ação", "Remover Outliers", "Remover Outliers Severos", "Substituir por Limites", "Substituir por Média", "Substituir por Mediana"],
            index=["Sem Ação", "Remover Outliers", "Remover Outliers Severos", "Substituir por Limites", "Substituir por Média", "Substituir por Mediana"].index(
                st.session_state.outlier_treatment_state[col]
            ),
            key=f"outlier_method_{col}_{len(st.session_state.treated_columns)}"
        )

        # Botão para aplicar o tratamento selecionado
        if st.button(f"Aplicar tratamento em {col}"):
            apply_outlier_treatment(col, method, st.session_state.initial_limits[col]["lower_bound"], st.session_state.initial_limits[col]["upper_bound"])
            if col not in st.session_state.treated_columns:
                st.session_state.treated_columns.append(col)
            st.rerun()

    # -------------------------------------
    # 📌 Boxplot Final Após Tratamento
    # -------------------------------------

    st.write("### Boxplot Após Tratamento")
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.data.boxplot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------------
    # 📌 Botão para Avançar para a Próxima Etapa
    # -------------------------------------

    if st.button("Próxima etapa"):
        st.session_state.step = 'data_summary'
        st.rerun()

# -------------------------------------
# 📌 FUNÇÃO DE SUGESTÃO AUTOMÁTICA PARA TRATAMENTO DE OUTLIERS
# -------------------------------------

def auto_select_outlier_treatment(col, data, lower_bound, upper_bound):
    """
    Sugere automaticamente o melhor método de tratamento de outliers com base na distribuição dos dados.

    Parâmetros:
    - col: Nome da coluna a ser analisada.
    - data: DataFrame contendo os dados.
    - lower_bound: Limite inferior dos valores considerados normais (IQR 1.5x abaixo do Q1).
    - upper_bound: Limite superior dos valores considerados normais (IQR 1.5x acima do Q3).

    Retorna:
    - Método sugerido para tratamento dos outliers.
    """

    # -------------------------------------
    # 📌 Cálculo da Proporção de Outliers
    # -------------------------------------

    total = len(data)  # Número total de registos

    # Contar outliers normais (fora do intervalo de 1.5 * IQR)
    total_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])

    # Contar outliers severos (fora do intervalo de 3 * IQR)
    total_severe_outliers = len(data[(data[col] < (lower_bound - 1.5 * (upper_bound - lower_bound))) |
                                     (data[col] > (upper_bound + 1.5 * (upper_bound - lower_bound)))])

    # Calcular percentagens
    percentage = total_outliers / total  # Percentagem de outliers normais
    severe_percentage = total_severe_outliers / total  # Percentagem de outliers severos

    # -------------------------------------
    # 📌 Verificação da Assimetria dos Dados (Skewness)
    # -------------------------------------

    skewness = data[col].skew()  # Medida de assimetria da distribuição dos dados

    # -------------------------------------
    # 📌 Definição das Regras para Sugerir o Melhor Método
    # -------------------------------------

    if severe_percentage > 0.10:
        # Se mais de 10% dos valores forem outliers severos, recomenda-se remover apenas os extremos
        return "Remover Outliers Severos"
    elif percentage > 0.20:
        # Se mais de 20% dos valores forem outliers, recomenda-se remover todos os outliers
        return "Remover Outliers"
    elif percentage > 0.05:
        # Se entre 5% e 20% forem outliers, recomenda-se substituí-los pelos limites aceitáveis
        return "Substituir por Limites"
    else:
        # Se houver menos de 5% de outliers, a escolha entre média e mediana é baseada na simetria
        if abs(skewness) > 1:
            return "Substituir por Mediana"  # Se houver alta assimetria, usa-se a mediana
        else:
            return "Substituir por Média"  # Caso contrário, a média é uma escolha razoável

# -------------------------------------
# 📌 FUNÇÃO PARA APLICAR TRATAMENTO DE OUTLIERS
# -------------------------------------

def apply_outlier_treatment(col, method, lower_bound, upper_bound):
    """
    Aplica o tratamento de outliers na coluna especificada, conforme o método escolhido.

    Parâmetros:
    - col: Nome da coluna a ser tratada.
    - method: Método de tratamento selecionado.
    - lower_bound: Limite inferior considerado aceitável.
    - upper_bound: Limite superior considerado aceitável.
    """

    # Obter os dados do estado global
    data = st.session_state.data

    # -------------------------------------
    # 📌 Remover Todos os Outliers (Fora do Intervalo 1.5 * IQR)
    # -------------------------------------
    
    if method == "Remover Outliers":
        st.session_state.data = data[
            (data[col] >= lower_bound) & (data[col] <= upper_bound)
        ]
        st.success(f"Todos os outliers removidos na coluna '{col}'.")

    # -------------------------------------
    # 📌 Remover Apenas Outliers Severos (Fora do Intervalo 3 * IQR)
    # -------------------------------------

    elif method == "Remover Outliers Severos":
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definir limites mais rigorosos para outliers severos (3 * IQR)
        severe_lower = Q1 - 3.0 * IQR
        severe_upper = Q3 + 3.0 * IQR

        st.session_state.data = data[
            (data[col] >= severe_lower) & (data[col] <= severe_upper)
        ]
        st.success(f"Outliers severos removidos na coluna '{col}'.")

    # -------------------------------------
    # 📌 Substituir Outliers pelos Limites Aceitáveis
    # -------------------------------------

    elif method == "Substituir por Limites":
        st.session_state.data[col] = data[col].clip(lower_bound, upper_bound)
        st.success(f"Valores substituídos pelos limites na coluna '{col}'.")

    # -------------------------------------
    # 📌 Substituir Outliers pela Média da Coluna
    # -------------------------------------

    elif method == "Substituir por Média":
        mean_value = data[col].mean()
        mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        st.session_state.data.loc[mask, col] = mean_value
        st.success(f"Valores substituídos pela média ({mean_value:.2f}) na coluna '{col}'.")

    # -------------------------------------
    # 📌 Substituir Outliers pela Mediana da Coluna
    # -------------------------------------

    elif method == "Substituir por Mediana":
        median_value = data[col].median()
        mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        st.session_state.data.loc[mask, col] = median_value
        st.success(f"Valores substituídos pela mediana ({median_value:.2f}) na coluna '{col}'.")


##########################################################
# -------------------------------------
# 📌 FUNÇÃO PARA GUARDAR O DATASET APÓS O PRÉ-PROCESSAMENTO
# -------------------------------------

def save_modified_dataset_in_memory():
    """
    Salva o dataset tratado na memória (session_state) para uso posterior.
    """

    # Criar uma cópia do dataset tratado e armazená-lo no estado da sessão
    st.session_state.data_tratada = st.session_state.data.copy()

    # Exibir uma mensagem de sucesso
    st.success("O dataset tratado foi salvo na memória para uso posterior.")

# -------------------------------------
# 📌 FUNÇÃO PARA PERMITIR O DOWNLOAD DO DATASET TRATADO
# -------------------------------------

def download_button(df, filename="dataset_tratado.csv"):
    """
    Permite ao utilizador descarregar o dataset tratado em formato CSV.

    Parâmetros:
    - df: DataFrame tratado a ser disponibilizado para download.
    - filename: Nome do ficheiro CSV a ser descarregado (padrão: "dataset_tratado.csv").
    """

    # Converter o DataFrame para formato CSV (sem índice)
    csv = df.to_csv(index=False)

    # Criar um buffer de memória para armazenar o conteúdo do ficheiro
    buf = io.BytesIO()

    # Escrever o conteúdo do CSV no buffer e posicionar o cursor no início
    buf.write(csv.encode())  # Converter para bytes e armazenar no buffer
    buf.seek(0)  # Definir a posição do cursor para o início do ficheiro

    # Criar um botão de download no Streamlit
    st.download_button(
        label="Baixar Dataset Tratado",  # Texto do botão
        data=buf,  # Ficheiro a ser descarregado
        file_name=filename,  # Nome do ficheiro ao fazer o download
        mime="text/csv"  # Tipo MIME do ficheiro
    )


##########################################################
# -------------------------------------
# 📌 CLASSE PARA CRIAR O PDF COM O RESUMO APÓS O PRÉ-PROCESSAMENTO
# -------------------------------------

from fpdf import FPDF
import requests
import tempfile
from datetime import datetime

class CustomPDF(FPDF):
    """
    Classe personalizada para gerar um relatório em PDF com cabeçalho e rodapé customizados.
    """

    def header(self):
        """
        Método para gerar o cabeçalho do PDF, incluindo o logótipo da instituição.
        """

        # URL do logótipo da instituição
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'

        # Fazer o download da imagem
        response = requests.get(logo_url)

        if response.status_code == 200:
            # Criar um ficheiro temporário para armazenar a imagem baixada
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                tmpfile.write(response.content)  # Escrever o conteúdo da imagem no ficheiro temporário
                tmpfile_path = tmpfile.name  # Obter o caminho do ficheiro

                # Adicionar a imagem no cabeçalho do PDF
                self.image(tmpfile_path, x=10, y=8, w=20)  # Definir posição e tamanho da imagem
        else:
            # Se a imagem não for baixada corretamente, exibir mensagem no PDF
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, "Logo não disponível", align='C')

        # Definir a fonte do cabeçalho
        self.set_font('Arial', 'B', 12)

        # Adicionar o título da plataforma no cabeçalho
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', align='C', ln=True)

        # Criar um espaço entre o cabeçalho e o conteúdo
        self.ln(15)

    def footer(self):
        """
        Método para gerar o rodapé do PDF, incluindo a data e número da página.
        """

        # Definir a posição do rodapé a 1.5 cm do final da página
        self.set_y(-15)

        # Definir a fonte do rodapé
        self.set_font('Arial', 'I', 10)

        # Obter a data atual no formato dia/mês/ano
        current_date = datetime.now().strftime('%d/%m/%Y')

        # Adicionar rodapé com a data e o número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', align='C')

# -------------------------------------
# 📌 FUNÇÃO PARA GERAR O PDF COM O RESUMO DO PRÉ-PROCESSAMENTO
# -------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def generate_pdf_resumo(dataset, summary_df, missing_data, outlier_summary):
    """
    Gera um relatório em PDF com informações estatísticas do dataset, valores ausentes, outliers,
    matriz de correlação e boxplot.

    Parâmetros:
    - dataset: DataFrame original após pré-processamento.
    - summary_df: DataFrame com estatísticas descritivas do dataset.
    - missing_data: Série contendo a contagem de valores ausentes por coluna.
    - outlier_summary: Lista contendo o resumo dos outliers identificados.

    Retorna:
    - Um buffer de memória contendo o PDF gerado.
    """

    # -------------------------------------
    # 📌 Função Auxiliar para Limpar Texto
    # -------------------------------------

    def clean_text(text):
        """Remove caracteres incompatíveis com a codificação do PDF."""
        if not isinstance(text, str):
            return text
        return text.encode('latin-1', errors='ignore').decode('latin-1')

    # -------------------------------------
    # 📌 Inicialização do PDF
    # -------------------------------------

    pdf = CustomPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=8)

    # -------------------------------------
    # 📌 Título do Relatório
    # -------------------------------------

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Relatório Resumo dos Dados"), ln=True, align="C")
    pdf.ln(5)

    # -------------------------------------
    # 📌 Estatísticas Descritivas Simplificadas
    # -------------------------------------

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Estatísticas Descritivas"), ln=True)
    pdf.set_font("Arial", size=8)

    # Criar DataFrame simplificado com estatísticas principais
    summary_simplified = pd.DataFrame({
        'Coluna': dataset.columns,
        'Tipo de Dados': dataset.dtypes,
        'Count': dataset.count(),
        'Top': dataset.mode().iloc[0],  # Valor mais frequente (moda)
    })

    # Inicializar colunas estatísticas apenas para colunas numéricas
    summary_simplified['std'] = None
    summary_simplified['min'] = None
    summary_simplified['max'] = None
    summary_simplified['Média'] = None

    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'Média'] = dataset[numeric_columns].mean()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'std'] = dataset[numeric_columns].std()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'min'] = dataset[numeric_columns].min()
    summary_simplified.loc[summary_simplified['Coluna'].isin(numeric_columns), 'max'] = dataset[numeric_columns].max()

    # Formatar valores numéricos para 4 casas decimais
    for col in ['Média', 'std', 'min', 'max']:
        summary_simplified[col] = summary_simplified[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Substituir 'nan' por vazio
    summary_simplified = summary_simplified.fillna('')

    # -------------------------------------
    # 📌 Adicionar Tabela das Estatísticas ao PDF
    # -------------------------------------

    pdf.set_fill_color(144, 238, 144)  # Cor de fundo do cabeçalho
    col_widths = [pdf.get_string_width(col) for col in summary_simplified.columns]
    max_width = 180

    total_width = sum(col_widths)
    scale_factor = max_width / total_width
    col_widths = [width * scale_factor for width in col_widths]

    for i, col in enumerate(summary_simplified.columns):
        pdf.cell(col_widths[i], 10, clean_text(col), 1, 0, 'C', True)
    pdf.ln()

    for i, row in summary_simplified.iterrows():
        for j, cell in enumerate(row):
            pdf.cell(col_widths[j], 8, clean_text(str(cell)), 1, 0, 'C')
        pdf.ln()

    pdf.ln(10)

    # -------------------------------------
    # 📌 Resumo de Valores Ausentes
    # -------------------------------------

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Resumo de Valores Ausentes"), ln=True)
    pdf.set_font("Arial", size=8)

    if not missing_data.empty:
        pdf.set_fill_color(144, 238, 144)
        pdf.cell(50, 10, clean_text("Variável"), 1, 0, 'C', True)
        pdf.cell(50, 10, clean_text("Total de Ausentes"), 1, 1, 'C', True)
        for col, count in missing_data.items():
            pdf.cell(50, 10, clean_text(col), 1)
            pdf.cell(50, 10, clean_text(str(count)), 1, 1)
        pdf.ln(10)
    else:
        pdf.cell(0, 10, txt=clean_text("Não há valores ausentes."), ln=True)

    # -------------------------------------
    # 📌 Resumo de Outliers
    # -------------------------------------

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Resumo de Outliers"), ln=True)
    pdf.set_font("Arial", size=8)

    if outlier_summary:
        pdf.set_fill_color(144, 238, 144)
        pdf.cell(50, 10, clean_text("Variável"), 1, 0, 'C', True)
        pdf.cell(50, 10, clean_text("Total de Outliers"), 1, 1, 'C', True)
        for entry in outlier_summary:
            pdf.cell(50, 10, clean_text(entry["Variável"]), 1)
            pdf.cell(50, 10, clean_text(str(entry["Total de Outliers"])), 1, 1)
        pdf.ln(10)
    else:
        pdf.cell(0, 10, txt=clean_text("Não há outliers."), ln=True)

    # -------------------------------------
    # 📌 Matriz de Correlação (Heatmap)
    # -------------------------------------

    pdf.cell(0, 10, txt=clean_text("Matriz de Correlação das Variáveis"), ln=True)
    numeric_data = dataset.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", cbar=True, square=True)
    plt.title('Matriz de Correlação das Variáveis', fontsize=14, fontweight='bold')

    temp_filename = "correlation_heatmap.png"
    plt.savefig(temp_filename)
    plt.close()
    pdf.image(temp_filename, x=10, w=180)
    pdf.ln(95)

    # -------------------------------------
    # 📌 Boxplot das Variáveis Numéricas
    # -------------------------------------

    pdf.cell(0, 10, txt=clean_text("Boxplot das Variáveis Numéricas"), ln=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_data)
    plt.title('Boxplot das Variáveis Numéricas')

    temp_filename_boxplot = "boxplot_combined.png"
    plt.savefig(temp_filename_boxplot)
    plt.close()
    pdf.image(temp_filename_boxplot, x=10, w=180)
    pdf.ln(75)

    # -------------------------------------
    # 📌 Gerar o PDF no Buffer de Memória
    # -------------------------------------

    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1', errors='ignore')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# -------------------------------------
# 📌 FUNÇÃO PARA SALVAR UMA TABELA COMO IMAGEM (PNG)
# -------------------------------------

import matplotlib.pyplot as plt

def save_table_as_image(df, filename="table_image.png"):
    """
    Converte um DataFrame Pandas numa imagem (PNG), formatando os valores para melhor visualização.

    Parâmetros:
    - df: DataFrame contendo a tabela a ser convertida em imagem.
    - filename: Nome do ficheiro da imagem a ser salva (padrão: "table_image.png").
    """

    # -------------------------------------
    # 📌 Tratamento de Valores no DataFrame Antes da Geração da Imagem
    # -------------------------------------

    # Substituir valores `NaN` por valores vazios para evitar exibições incorretas
    df = df.fillna('')

    # Formatar valores numéricos para 4 casas decimais
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # -------------------------------------
    # 📌 Configuração da Figura para Geração da Tabela
    # -------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))  # Define o tamanho da imagem gerada
    ax.axis('tight')  # Ajusta os limites para caber na figura
    ax.axis('off')  # Remove os eixos para melhor visualização

    # Criar a tabela no gráfico
    table = ax.table(
        cellText=df.values,  # Conteúdo da tabela
        colLabels=df.columns,  # Cabeçalhos das colunas
        loc='center',  # Centralizar a tabela na imagem
        cellLoc='center',  # Centralizar o texto nas células
        colColours=['#D9EAF7'] * len(df.columns)  # Definir cor do cabeçalho da tabela
    )

    # -------------------------------------
    # 📌 Ajustes de Formatação da Tabela
    # -------------------------------------

    table.auto_set_font_size(False)  # Desativar ajuste automático do tamanho da fonte
    table.set_fontsize(10)  # Definir tamanho da fonte manualmente
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Ajustar automaticamente a largura das colunas

    # -------------------------------------
    # 📌 Salvamento da Tabela Como Imagem (PNG)
    # -------------------------------------

    plt.savefig(filename, format='png', bbox_inches='tight')  # Salvar imagem no formato PNG
    plt.close()  # Fechar a figura para evitar sobrecarga de memória

# Resumo do Pré-processamento de dados:
# -------------------------------------
# 📌 FUNÇÃO PARA GERAR O RESUMO DOS DADOS
# -------------------------------------

def data_summary():
    """
    Apresenta um resumo dos dados tratados, incluindo estatísticas descritivas, valores ausentes,
    detecção de outliers, boxplots e matriz de correlação. Além disso, permite o download do resumo
    em PDF e do dataset tratado.
    """

    st.subheader("Resumo dos Dados")

    # -------------------------------------
    # 📌 Verificar Disponibilidade do Dataset
    # -------------------------------------

    if 'data' in st.session_state and st.session_state.data is not None:
        dataset = st.session_state.data
        st.success("Usando o dataset tratado!")
    else:
        st.error("Nenhum dataset está disponível. Por favor, execute o tratamento de dados antes.")
        return  # Encerra a função caso não haja dados disponíveis

    # -------------------------------------
    # 📌 Seleção de Colunas para Exibição
    # -------------------------------------

    # Obter colunas selecionadas ou usar todas as colunas do dataset
    selected_columns = st.session_state.get('selected_columns', [])
    if not selected_columns:
        selected_columns = dataset.columns.tolist()

    # Permitir que o utilizador selecione as colunas para visualização
    selected_columns_to_display = st.multiselect(
        "Selecione as variáveis para visualizar as estatísticas",
        options=selected_columns,
        default=selected_columns
    )

    # Exibir o número de linhas e colunas do dataset filtrado
    st.write("Número de linhas e colunas:", dataset[selected_columns_to_display].shape)

    # -------------------------------------
    # 📌 Estatísticas Descritivas
    # -------------------------------------

    # Identificar colunas numéricas
    numeric_columns = dataset[selected_columns_to_display].select_dtypes(include=['number']).columns

    # Criar um dicionário para armazenar estatísticas
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

    # Converter para DataFrame e adicionar os tipos de dados
    summary_df = pd.DataFrame(summary_data)
    summary_df['Tipo de Dados'] = dataset[selected_columns_to_display].dtypes

    # Arredondar valores numéricos para 4 casas decimais e preencher valores ausentes com 0
    summary_df = summary_df.round(4).fillna(0)

    # Exibir a tabela de estatísticas descritivas
    st.write("Estatísticas Descritivas e Tipos de Dados")
    st.dataframe(fix_dataframe_types(summary_df))

    # -------------------------------------
    # 📌 Análise de Valores Ausentes
    # -------------------------------------

    st.subheader("Resumo de Valores Ausentes")

    # Identificar colunas com valores ausentes
    missing_data = dataset[selected_columns_to_display].isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if not missing_data.empty:
        st.write("Valores ausentes encontrados:")
        st.dataframe(fix_dataframe_types(missing_data.rename("Total de Valores Ausentes")))
    else:
        st.write("Não há valores ausentes nas variáveis selecionadas.")

    # -------------------------------------
    # 📌 Análise de Outliers
    # -------------------------------------

    st.subheader("Resumo de Outliers")

    # Selecionar apenas colunas numéricas
    numeric_data = dataset[selected_columns_to_display].select_dtypes(include=['number'])

    # Obter colunas já tratadas
    treated_columns = st.session_state.get('treated_columns', [])

    # Criar lista para armazenar o resumo dos outliers
    outlier_summary = []

    if not numeric_data.empty:
        for column in numeric_data.columns:
            if column in treated_columns:  # Ignorar colunas já tratadas
                continue

            # Cálculo dos quartis e do intervalo interquartil (IQR)
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identificar outliers
            outliers = numeric_data[(numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)]
            if len(outliers) > 0:
                outlier_summary.append({
                    "Variável": column,
                    "Total de Outliers": len(outliers)
                })

        # Exibir o resumo dos outliers encontrados
        if outlier_summary:
            st.dataframe(fix_dataframe_types(pd.DataFrame(outlier_summary)))
        else:
            st.write("Não há outliers nas variáveis selecionadas.")
    else:
        st.write("Nenhuma variável numérica para análise de outliers.")

    # -------------------------------------
    # 📌 Gráfico Boxplot das Variáveis Numéricas
    # -------------------------------------

    st.subheader("Boxplot das Variáveis Numéricas")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_data)
    plt.title('Boxplot das Variáveis Numéricas')
    st.pyplot(plt)

    # -------------------------------------
    # 📌 Matriz de Correlação (Heatmap)
    # -------------------------------------

    st.subheader("Matriz de Correlação das Variáveis")

    # Calcular a correlação entre variáveis numéricas
    correlation_matrix = numeric_data.corr()

    # Gerar e exibir o heatmap da correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", cbar=True, square=True)
    plt.title('Matriz de Correlação das Variáveis', fontsize=14, fontweight='bold', fontname='Arial')
    st.pyplot(plt)

    # -------------------------------------
    # 📌 Download do Resumo em PDF
    # -------------------------------------

    pdf_buffer = generate_pdf_resumo(dataset, summary_df, missing_data, outlier_summary)
    st.download_button(
        label="Baixar PDF com o Resumo",
        data=pdf_buffer,
        file_name="resumo_dos_dados.pdf",
        mime="application/pdf"
    )

    # -------------------------------------
    # 📌 Download do Dataset Tratado
    # -------------------------------------

    dataset_to_download = dataset[selected_columns_to_display]
    download_button(dataset_to_download)

    # -------------------------------------
    # 📌 Navegação Entre Etapas
    # -------------------------------------

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
# -------------------------------------
# 📌 FUNÇÃO PARA PLOTAR MÉTRICAS DE DESEMPENHO DOS MODELOS
# -------------------------------------

import streamlit as st
import matplotlib.pyplot as plt

def plot_metrics(metrics_df):
    """
    Gera gráficos para visualizar as métricas de desempenho dos modelos, diferenciando entre
    tarefas de classificação e regressão.

    Parâmetros:
    - metrics_df: DataFrame contendo as métricas de desempenho dos modelos.

    Retorno:
    - Exibe os gráficos no Streamlit.
    """

    try:
        # -------------------------------------
        # 📌 Inicializar Armazenamento de Métricas no Estado da Sessão
        # -------------------------------------

        # Se a chave 'metrics' ainda não estiver no session_state, inicializá-la
        if 'metrics' not in st.session_state:
            st.session_state['metrics'] = {}

        # Verificar se o DataFrame está vazio
        if metrics_df.empty:
            st.warning("Nenhum dado para exibir no gráfico.")
            return

        # Armazenar as métricas no estado da sessão para referência posterior
        for _, row in metrics_df.iterrows():
            model_name = row.name  # Assumindo que o índice contém o nome do modelo
            st.session_state['metrics'][model_name] = row.to_dict()

        # -------------------------------------
        # 📌 Configuração do Índice e Identificação do Tipo de Modelo
        # -------------------------------------

        # Definir a coluna 'Modelo' como índice, se ainda não estiver
        metrics_df.set_index('Modelo', inplace=True)

        # Listas de métricas típicas para classificação e regressão
        classification_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        regression_columns = ['MSE', 'MAE', 'R²']

        # -------------------------------------
        # 📌 Plotagem de Gráficos de Classificação
        # -------------------------------------

        if all(col in metrics_df.columns for col in classification_columns):
            # Criar a figura do gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plotar as métricas de classificação
            metrics_df[classification_columns].plot(kind='bar', ax=ax)

            # Configuração do gráfico
            plt.title('Métricas de Desempenho dos Modelos (Classificação)', fontsize=16)
            plt.ylabel('Valor', fontsize=14)
            plt.xlabel('Modelos', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylim(0, 1)  # As métricas de classificação geralmente variam entre 0 e 1
            plt.legend(loc='lower right', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        # -------------------------------------
        # 📌 Plotagem de Gráficos de Regressão
        # -------------------------------------

        elif all(col in metrics_df.columns for col in regression_columns):
            # Criar a figura do gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plotar as métricas de regressão
            metrics_df[regression_columns].plot(kind='bar', ax=ax)

            # Configuração do gráfico
            plt.title('Métricas de Desempenho dos Modelos (Regressão)', fontsize=16)
            plt.ylabel('Valor', fontsize=14)
            plt.xlabel('Modelos', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        else:
            st.error("O DataFrame não contém métricas válidas para classificação ou regressão.")
            return  # Se não há métricas válidas, encerra a função

        # -------------------------------------
        # 📌 Exibir o Gráfico no Streamlit
        # -------------------------------------

        st.pyplot(fig)

    except Exception as e:
        # Tratamento de erros genérico para evitar falhas inesperadas
        st.error(f"Ocorreu um erro ao plotar as métricas: {str(e)}")

    finally:
        # Limpar a figura para evitar sobreposição de gráficos na interface do Streamlit
        plt.clf()

# -------------------------------------
# 📌 FUNÇÃO PARA DEFINIR O GRID DE HIPERPARÂMETROS PADRÃO PARA CADA MODELO
# -------------------------------------

def get_default_param_grid(model_name):
    """
    Retorna um dicionário contendo os hiperparâmetros padrão para cada modelo de Machine Learning.

    Parâmetros:
    - model_name: Nome do modelo para o qual se deseja obter o conjunto de hiperparâmetros.

    Retorno:
    - Dicionário com os hiperparâmetros e os respetivos intervalos de valores para otimização.
    """

    # -------------------------------------
    # 📌 Configuração do Grid Search para Support Vector Classification (SVC)
    # -------------------------------------
    if model_name == "Support Vector Classification (SVC)":
        return {
            'C': [0.1, 1, 10],  # Define a penalização do erro
            'kernel': ['linear', 'rbf'],  # Tipos de kernel utilizados
            'gamma': ['scale', 'auto']  # Apenas utilizado quando kernel='rbf'
        }

    # -------------------------------------
    # 📌 Configuração do Grid Search para K-Nearest Neighbors (KNN)
    # -------------------------------------
    elif model_name == "K-Nearest Neighbors (KNN)":
        return {
            'n_neighbors': list(range(1, 21)),  # Testa todos os valores de 1 a 20 para o número de vizinhos
            'weights': ['uniform', 'distance']  # Define a forma de ponderação das distâncias
        }

    # -------------------------------------
    # 📌 Configuração do Grid Search para Random Forest
    # -------------------------------------
    elif model_name == "Random Forest":
        # Geração dinâmica do parâmetro `max_depth`
        max_depth_range = [None] + list(range(5, 21, 5))  # [None, 5, 10, 15, 20]
        return {
            'max_depth': max_depth_range,  # Profundidade máxima da árvore
            'n_estimators': [10, 50, 100]  # Número de árvores na floresta
        }

    # -------------------------------------
    # 📌 Configuração do Grid Search para Suporte de Vetores em Regressão (SVR)
    # -------------------------------------
    elif model_name == "Regressão por Vetores de Suporte (SVR)":
        return {
            'C': [1, 10],  # Penalização do erro
            'epsilon': [0.1, 0.2],  # Margem de tolerância para erro
            'kernel': ['linear', 'rbf']  # Tipos de kernel utilizados
        }

    # -------------------------------------
    # 📌 Configuração para Regressão Linear Simples (RLS)
    # -------------------------------------
    elif model_name == "Regressão Linear Simples (RLS)":
        return {}  # A regressão linear geralmente não requer ajuste de hiperparâmetros

    # -------------------------------------
    # 📌 Retorno para modelos não especificados
    # -------------------------------------
    else:
        return {}  # Se o modelo não for reconhecido, retorna um dicionário vazio

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

                # Mostrar intervalo aceito como dica para o utilizador
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


# Função para configurar a validação cruzada com base na escolha do utilizador
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

# Treina um modelo de Regressão por Vetores de Suporte (SVR) com GridSearch opcional
def train_svr_with_gridsearch(X_train, y_train, X_test, y_test, use_grid_search=True, manual_params=None):
    """
    Train Support Vector Regression (SVR) model with optional GridSearchCV
    
    Parameters:
    -----------
    X_train : array-like
        Training feature matrix
    y_train : array-like
        Training target vector
    X_test : array-like
        Testing feature matrix
    y_test : array-like
        Testing target vector
    use_grid_search : bool, optional (default=True)
        Whether to use GridSearchCV for hyperparameter tuning
    manual_params : dict, optional
        Manually specified parameters to override GridSearch
    
    Returns:
    --------
    dict
        Dictionary containing model performance metrics and details
    """
    try:
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Base SVR model
        svr = SVR()
        
        # Default parameter grid for SVR
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        # If manual parameters are provided, update the param_grid
        if manual_params:
            for param, value in manual_params.items():
                # Ensure the value is a list for GridSearchCV
                param_grid[param] = [value] if not isinstance(value, list) else value
        
        # Cross-validation strategy
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        
        if use_grid_search:
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=svr, 
                param_grid=param_grid, 
                cv=cv_strategy, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Best model from GridSearch
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Use manual or default parameters
            if manual_params:
                svr.set_params(**manual_params)
            
            best_model = svr.fit(X_train_scaled, y_train)
            best_params = manual_params or {}
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Prepare metrics dictionary
        metrics = {
            "Modelo": "Support Vector Regression (SVR)",
            "R²": r2,
            "MAE": mae,
            "MSE": mse,
            "Best Parameters": best_params
        }
        
        return metrics
    
    except Exception as e:
        st.error(f"Erro ao treinar o modelo SVR: {str(e)}")
        return None

def train_model_with_gridsearch(model, param_grid, X_train, y_train, use_grid_search, manual_params=None, cv_choice="K-Fold"):
    try:
        # Inicializar parâmetros manuais como vazio, se não fornecido
        if manual_params is None:
            manual_params = {}

        # Obter o nome do modelo
        model_name = type(model).__name__

        # Logs para diagnóstico - Parâmetros no estado global antes do treino
        st.write("Parâmetros no estado global antes do treino:")
        st.write("best_params:", st.session_state.get('best_params', {}))
        st.write("manual_params:", st.session_state.get('manual_params', {}))

        # Carregar parâmetros salvos do estado global
        saved_params = st.session_state.get('best_params', None)

        # Aplicar parâmetros salvos, se existirem e não usar GridSearch
        if saved_params and not use_grid_search:
            st.info(f"Aplicando parâmetros salvos ao modelo: {saved_params}")
            model.set_params(**saved_params)

        # Remover 'gamma' se o kernel for 'linear'
        if manual_params.get("kernel") == "linear" and "gamma" in manual_params:
            del manual_params["gamma"]
            if 'gamma' in st.session_state.get('manual_params', {}):
                del st.session_state['manual_params']['gamma']

        # Se usar GridSearch
        if use_grid_search:
            # Atualizar grid com parâmetros manuais fornecidos
            if manual_params:
                for param, value in manual_params.items():
                    if not isinstance(value, list):
                        manual_params[param] = [value]
                param_grid.update(manual_params)

            # Configurar validação cruzada
            cv_strategy = get_cv_strategy(cv_choice, X_train, y_train)
            scoring = 'r2' if model_name == "SVR" else 'accuracy'

            # Treinar com GridSearch
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Melhor modelo e parâmetros
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            st.session_state['best_params'] = best_params
            st.success(f"Melhores parâmetros encontrados: {best_params}")

            return best_model, best_params

        else:
            # Se não usar GridSearch, aplicar manualmente os parâmetros
            valid_params = model.get_params().keys()
            manual_params = {k: v for k, v in manual_params.items() if k in valid_params}
            model.set_params(**manual_params)

            # Treinar diretamente
            model.fit(X_train, y_train)

            # Salvar parâmetros manuais no estado global
            st.session_state['manual_params'] = manual_params
            st.success(f"Parâmetros manuais salvos: {manual_params}")

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
                "Clustering Hierárquico": AgglomerativeClustering(linkage='ward'),
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
    from sklearn.decomposition import PCA
    import numpy as np

    # Inicializar a variável `best_n_clusters_retrain` com um valor padrão
    best_n_clusters_retrain = None

    # Inicializar estados se não existirem
    if 'pca_configured' not in st.session_state:
        st.session_state.pca_configured = False
    if 'ready_for_clustering' not in st.session_state:
        st.session_state.ready_for_clustering = False

    # Função para a configuração de Clustering
    if st.session_state.model_type == "Clustering" and st.session_state.selected_model_name:
        st.write("### Configuração para Clustering")

        # Dados categóricos codificados
        X = pd.get_dummies(st.session_state.data)

        # Padronizar os dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ETAPA 1: Configuração do PCA para Clustering Hierárquico
        if st.session_state.selected_model_name == "Clustering Hierárquico" and not st.session_state.pca_configured:
            st.write("### Redução de Dimensionalidade com PCA para Clustering Hierárquico")
            
            # Verificar se o dataset é grande o suficiente para um aviso
            if X.shape[0] > 1000 or X.shape[1] > 10:
                st.warning(f"Atenção: Seu dataset tem {X.shape[0]} registros e {X.shape[1]} dimensões. A aplicação de PCA é necessária para Clustering Hierárquico.")
            
            # Permitir ao utilizador escolher o número de componentes ou usar valor automático
            use_auto_components = st.checkbox("Determinar automaticamente o número de componentes", value=True, key="auto_comp_hierarch")
            
            if use_auto_components:
                # Calcular o PCA para determinar a variância explicada
                pca_full = PCA().fit(X_scaled)
                explained_variance_ratio = pca_full.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                # Encontrar o número de componentes que explicam pelo menos 90% da variância
                n_components = np.argmax(cumulative_variance >= 0.9) + 1
                n_components = min(n_components, 10)  # Limitar a no máximo 10 componentes
                
                st.write(f"Número de componentes selecionados automaticamente: {n_components} (explica aproximadamente {cumulative_variance[n_components-1]*100:.1f}% da variância)")
                
                # Mostrar gráfico de variância explicada
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
                ax.axhline(y=0.9, color='r', linestyle='--', label='90% Variância Explicada')
                ax.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} Componentes')
                ax.set_xlabel('Número de Componentes')
                ax.set_ylabel('Variância Explicada Acumulada')
                ax.set_title('Variância Explicada por Componentes do PCA')
                ax.legend()
                st.pyplot(fig)
                plt.clf()
            else:
                # Permitir que o utilizador escolha o número de componentes
                max_components = min(X.shape[1], 20)  # Limitar ao número de features ou 20, o que for menor
                n_components = st.slider("Número de componentes PCA para Hierárquico", 2, max_components, value=min(3, max_components), key="n_comp_hierarch")
            
            # Botão para confirmar a configuração do PCA
            if st.button("Confirmar Configuração do PCA para Clustering Hierárquico"):
                # Aplicar PCA com o número de componentes escolhido
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Salvar no estado da sessão
                st.session_state.X_pca = X_pca
                st.session_state.pca_n_components = n_components
                st.session_state.pca_configured = True
                st.session_state.pca_model = pca
                st.session_state.explained_variance = pca.explained_variance_ratio_
                
                st.success(f"PCA configurado com sucesso! Dimensionalidade reduzida de {X_scaled.shape[1]} para {X_pca.shape[1]} componentes.")
                
                # Visualização 2D dos dados com PCA se tivermos pelo menos 2 componentes
                if n_components >= 2:
                    st.write("### Visualização dos Dados Após PCA")
                    
                    # Permitir que o utilizador escolha quais componentes visualizar
                    available_components = min(n_components, 10)  # Limitar a 10 para evitar sobrecarga
                    
                    component_x = st.selectbox(
                        "Escolha o componente para o eixo X:",
                        options=list(range(available_components)),
                        format_func=lambda x: f"Componente {x+1}",
                        index=0,
                        key="comp_x_hierarch"
                    )
                    
                    component_y = st.selectbox(
                        "Escolha o componente para o eixo Y:",
                        options=list(range(available_components)),
                        format_func=lambda x: f"Componente {x+1}",
                        index=1 if available_components > 1 else 0,
                        key="comp_y_hierarch"
                    )
                    
                    # Criar a visualização 2D baseada nos componentes escolhidos
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(X_pca[:, component_x], X_pca[:, component_y], alpha=0.7)
                    ax.set_xlabel(f'Componente Principal {component_x+1}', fontsize=12)
                    ax.set_ylabel(f'Componente Principal {component_y+1}', fontsize=12)
                    ax.set_title(f'Visualização 2D dos Componentes PCA {component_x+1} e {component_y+1}', fontsize=14, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Mostrar a variância explicada por estes componentes
                    if hasattr(pca, 'explained_variance_ratio_'):
                        var_x = pca.explained_variance_ratio_[component_x] * 100
                        var_y = pca.explained_variance_ratio_[component_y] * 100
                        ax.set_xlabel(f'Componente Principal {component_x+1} ({var_x:.1f}% variância)', fontsize=12)
                        ax.set_ylabel(f'Componente Principal {component_y+1} ({var_y:.1f}% variância)', fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()

            # Botão para avançar para a configuração do clustering (fora do if anterior)
                if st.button("Prosseguir para Clustering"):
                    st.session_state.ready_for_clustering = True
                    st.rerun()
            
        # ETAPA 1: Configuração do PCA para KMeans
        if st.session_state.selected_model_name == "KMeans" and not st.session_state.pca_configured:
            st.write("### Redução de Dimensionalidade com PCA")
            
            # Verificar se o dataset é grande o suficiente para um aviso
            if X.shape[0] > 1000 or X.shape[1] > 10:
                st.warning(f"Atenção: Seu dataset tem {X.shape[0]} registros e {X.shape[1]} dimensões. A aplicação de PCA é altamente recomendada.")
            
            # Permitir ao utilizador escolher o número de componentes ou usar valor automático
            use_auto_components = st.checkbox("Determinar automaticamente o número de componentes", value=True)
            
            if use_auto_components:
                # Calcular o PCA para determinar a variância explicada
                pca_full = PCA().fit(X_scaled)
                explained_variance_ratio = pca_full.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                # Encontrar o número de componentes que explicam pelo menos 90% da variância
                n_components = np.argmax(cumulative_variance >= 0.9) + 1
                n_components = min(n_components, 10)  # Limitar a no máximo 10 componentes
                
                st.write(f"Número de componentes selecionados automaticamente: {n_components} (explica aproximadamente {cumulative_variance[n_components-1]*100:.1f}% da variância)")
                
                # Mostrar gráfico de variância explicada
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
                ax.axhline(y=0.9, color='r', linestyle='--', label='90% Variância Explicada')
                ax.axvline(x=n_components, color='g', linestyle='--', label=f'{n_components} Componentes')
                ax.set_xlabel('Número de Componentes')
                ax.set_ylabel('Variância Explicada Acumulada')
                ax.set_title('Variância Explicada por Componentes do PCA')
                ax.legend()
                st.pyplot(fig)
                plt.clf()
            else:
                # Permitir que o utilizador escolha o número de componentes
                max_components = min(X.shape[1], 20)  # Limitar ao número de features ou 20, o que for menor
                n_components = st.slider("Número de componentes PCA", 2, max_components, value=min(3, max_components))
            
            # Botão para confirmar a configuração do PCA
            if st.button("Confirmar Configuração do PCA"):
                # Aplicar PCA com o número de componentes escolhido
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Salvar no estado da sessão
                st.session_state.X_pca = X_pca
                st.session_state.pca_n_components = n_components
                st.session_state.pca_configured = True
                st.session_state.pca_model = pca
                st.session_state.explained_variance = pca.explained_variance_ratio_
                
                st.success(f"PCA configurado com sucesso! Dimensionalidade reduzida de {X_scaled.shape[1]} para {X_pca.shape[1]} componentes.")
                
                # Visualização 2D e 3D simultânea dos dados com PCA se tivermos pelo menos 2 componentes
                if n_components >= 2:
                    st.write("### Visualização dos Dados Após PCA")
                    
                    # Permitir que o utilizador escolha quais componentes visualizar
                    available_components = min(n_components, 10)  # Limitar a 10 para evitar sobrecarga
                    
                    component_x = st.selectbox(
                        "Escolha o componente para o eixo X:",
                        options=list(range(available_components)),
                        format_func=lambda x: f"Componente {x+1}",
                        index=0
                    )
                    
                    component_y = st.selectbox(
                        "Escolha o componente para o eixo Y:",
                        options=list(range(available_components)),
                        format_func=lambda x: f"Componente {x+1}",
                        index=1 if available_components > 1 else 0
                    )
                    
                    # Criar a visualização 2D baseada nos componentes escolhidos
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(X_pca[:, component_x], X_pca[:, component_y], alpha=0.7)
                    ax.set_xlabel(f'Componente Principal {component_x+1}', fontsize=12)
                    ax.set_ylabel(f'Componente Principal {component_y+1}', fontsize=12)
                    ax.set_title(f'Visualização 2D dos Componentes PCA {component_x+1} e {component_y+1}', fontsize=14, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Mostrar a variância explicada por estes componentes (se disponível)
                    if hasattr(pca, 'explained_variance_ratio_'):
                        var_x = pca.explained_variance_ratio_[component_x] * 100
                        var_y = pca.explained_variance_ratio_[component_y] * 100
                        ax.set_xlabel(f'Componente Principal {component_x+1} ({var_x:.1f}% variância)', fontsize=12)
                        ax.set_ylabel(f'Componente Principal {component_y+1} ({var_y:.1f}% variância)', fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()

                # Botão para avançar para a configuração do clustering
                if st.button("Prosseguir para Clustering"):
                    st.session_state.ready_for_clustering = True
                    st.rerun()
        
        # ETAPA 2: Configuração do Clustering (após o PCA para Hierarchical ou diretamente para K-means)
        elif st.session_state.selected_model_name == "KMeans" or (st.session_state.selected_model_name == "Clustering Hierárquico" and st.session_state.pca_configured):
            # Escolher o intervalo de clusters (reduzido de 2-20 para 2-10 por padrão para ser menos pesado)
            num_clusters_range = st.slider("Intervalo de clusters para explorar (para análise)", 2, 10, (2, 6))
            
            # Preparar dados para análise
            if st.session_state.selected_model_name == "Clustering Hierárquico":
                # Para clustering hierárquico, usar dados com PCA
                training_data = st.session_state.X_pca
            else:
                # Para K-means, usar dados originais
                training_data = X_scaled
            
            # Opção para usar amostragem para análise mais rápida
            use_sampling = st.checkbox("Usar amostragem dos dados para análise mais rápida", value=True)
            if use_sampling:
                sample_size = st.slider("Tamanho da amostra", 
                                    min_value=min(100, training_data.shape[0]),
                                    max_value=min(2000, training_data.shape[0]),
                                    value=min(1000, training_data.shape[0]))
                # Realizar amostragem
                np.random.seed(42)  # Para reprodutibilidade
                sample_indices = np.random.choice(training_data.shape[0], sample_size, replace=False)
                analysis_data = training_data[sample_indices]
                st.info(f"Usando {sample_size} pontos ({sample_size/training_data.shape[0]:.1%} dos dados) para análise.")
            else:
                analysis_data = training_data
            
            # Análise de clusters
            st.write("### Análise para Determinação do Número de Clusters")
            silhouette_scores = []
            davies_bouldin_scores = []
            calinski_harabasz_scores = []

            # Adicionar barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calcular métricas para cada número de clusters
            total_iterations = num_clusters_range[1] - num_clusters_range[0] + 1

            # Condicional para KMeans e Clustering Hierárquico
            for i, n_clusters in enumerate(range(num_clusters_range[0], num_clusters_range[1] + 1)):
                # Atualizar barra de progresso
                progress = (i + 1) / total_iterations
                progress_bar.progress(progress)
                status_text.text(f"Analisando com {n_clusters} clusters... ({i+1}/{total_iterations})")
                
                try:
                    if st.session_state.selected_model_name == "KMeans":
                        # Otimização: Reduzir n_init e max_iter para KMeans
                        temp_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100)
                    else:  # Clustering Hierárquico
                        temp_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    
                    # Treinar modelo com dados amostrados
                    temp_model.fit(analysis_data)
                    labels = temp_model.labels_
                    
                    # Calcular as métricas
                    silhouette_scores.append(silhouette_score(analysis_data, labels))
                    davies_bouldin_scores.append(davies_bouldin_score(analysis_data, labels))
                    calinski_harabasz_scores.append(calinski_harabasz_score(analysis_data, labels))
                    
                except Exception as e:
                    st.error(f"Erro ao processar {n_clusters} clusters: {str(e)}")
                    # Adicionar valores neutros para manter o array no tamanho correto
                    silhouette_scores.append(0)
                    davies_bouldin_scores.append(float('inf'))
                    calinski_harabasz_scores.append(0)
            
            # Limpar barra de progresso e status
            status_text.empty()
            progress_bar.empty()

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

            # Escolher o melhor número de clusters com base no Silhouette Score
            if silhouette_scores and any(score > 0 for score in silhouette_scores):
                best_n_clusters = metrics_df.loc[metrics_df["Silhouette Score"].idxmax(), "Número de Clusters"]
                st.write(f"**Melhor Número de Clusters** (com base no Silhouette Score): {best_n_clusters}")
                best_n_clusters_retrain = best_n_clusters

            # Escolher abordagem para número de clusters
            st.write("### Escolha a Abordagem para Determinar o Número de Clusters")
            method = st.radio("Selecione a abordagem:", ["Automático", "Manual"], key="initial_training_method")

            if method == "Automático":
                # Escolher o melhor número de clusters com base no Silhouette Score
                if silhouette_scores and any(score > 0 for score in silhouette_scores):
                    best_n_clusters = range(num_clusters_range[0], num_clusters_range[1] + 1)[np.argmax(silhouette_scores)]
                    best_n_clusters_retrain = best_n_clusters  # Atualizar o valor para re-treino
                else:
                    st.error("Não foi possível determinar automaticamente o número de clusters. Por favor, selecione manualmente.")
                    best_n_clusters_retrain = 3  # Valor padrão

            elif method == "Manual":
                best_n_clusters = st.slider("Escolha o número de clusters", num_clusters_range[0], num_clusters_range[1], value=3)
                best_n_clusters_retrain = best_n_clusters  # Atualizar o valor para re-treino

            # Garantir que `best_n_clusters_retrain` tenha um valor válido antes de usar
            if best_n_clusters_retrain is None:
                st.warning("Por favor, selecione uma abordagem para determinar o número de clusters.")
            else:
                # Treinar modelo inicial
                if st.button(f"Treinar Modelo Inicial"):
                    # Configurar e treinar o modelo (usando todos os dados para treino final)
                    if st.session_state.selected_model_name == "Clustering Hierárquico":
                        model = st.session_state.models["Clustering Hierárquico"]
                        model.set_params(n_clusters=best_n_clusters_retrain, linkage='ward')
                    else:  # KMeans
                        model = st.session_state.models["KMeans"]
                        # Otimizar KMeans para maior velocidade no treino final
                        model.set_params(n_clusters=best_n_clusters_retrain, n_init=5, max_iter=300)
                    
                    # Barra de progresso para o treino
                    with st.spinner(f"Treinando o modelo com {best_n_clusters_retrain} clusters..."):
                        model.fit(training_data)
                        st.session_state.clustering_labels = model.labels_
                    
                    # Calcular métricas
                    st.session_state.initial_metrics = {
                        "Número de Clusters": best_n_clusters_retrain,
                        "Silhouette Score": silhouette_score(training_data, st.session_state.clustering_labels),
                        "Davies-Bouldin Index": davies_bouldin_score(training_data, st.session_state.clustering_labels),
                        "Calinski-Harabasz Score": calinski_harabasz_score(training_data, st.session_state.clustering_labels)
                    }
                    
                    # Salvar informações importantes no estado da sessão
                    st.session_state.training_data = training_data
                    st.session_state.training_completed = True
                    st.session_state.trained_model = model  # Salvar o modelo treinado
                    
                    # Mostrar mensagem de sucesso
                    if st.session_state.selected_model_name == "Clustering Hierárquico":
                        st.success(f"Modelo hierárquico treinado com sucesso usando {best_n_clusters_retrain} clusters e {st.session_state.pca_n_components} componentes PCA!")
                    else:
                        st.success(f"Modelo K-means treinado com sucesso usando {best_n_clusters_retrain} clusters!")

            # Exibir métricas e próxima ação apenas após o treino
            if st.session_state.get("training_completed", False):
                st.write("### Métricas do Treino Inicial")
                st.table(fix_dataframe_types(pd.DataFrame([st.session_state.initial_metrics])))

                # Visualização dos clusters
                if 'clustering_labels' in st.session_state:
                    st.write("### Visualização dos Clusters")
                                        
                    # Para KMeans podemos mostrar os centroides
                    if st.session_state.selected_model_name == "KMeans":
                        if "trained_model" in st.session_state and hasattr(st.session_state.trained_model, 'cluster_centers_'):
                            st.write("#### Centroides dos Clusters")
                            centroids = st.session_state.trained_model.cluster_centers_
                            if centroids.shape[1] > 10:
                                st.write(f"(Mostrando apenas as primeiras 10 dimensões de {centroids.shape[1]})")
                                centroids_df = pd.DataFrame(centroids[:, :10])
                            else:
                                centroids_df = pd.DataFrame(centroids)
                            
                            st.dataframe(fix_dataframe_types(centroids_df))
                    
                    # Preparar dados para visualização
                    if st.session_state.selected_model_name == "Clustering Hierárquico":
                        # Para hierárquico, já temos os dados PCA
                        plot_data = st.session_state.X_pca
                    else:
                        # Para K-means, podemos reduzir os dados para visualização se necessário
                        if X_scaled.shape[1] > 3:
                            pca_viz = PCA(n_components=3)
                            plot_data = pca_viz.fit_transform(X_scaled)
                            st.write("(Dados reduzidos via PCA para visualização)")
                        else:
                            plot_data = X_scaled

                    # Obter número total de componentes
                    total_components = plot_data.shape[1]

                    # Permitir escolha de componentes para x e y
                    st.write("### Escolha os Componentes para Visualização")
                    col1, col2 = st.columns(2)

                    with col1:
                        x_component = st.selectbox(
                            "Componente para o Eixo X", 
                            list(range(total_components)), 
                            index=0,
                            format_func=lambda x: f"Componente {x+1}",
                            key="initial_x_component"  # Chave única adicionada
                        )

                    with col2:
                        y_component = st.selectbox(
                            "Componente para o Eixo Y", 
                            list(range(total_components)), 
                            index=1 if total_components > 1 else 0,
                            format_func=lambda x: f"Componente {x+1}",
                            key="initial_y_component"  # Chave única adicionada
                        )

                    # Verificar se componentes são diferentes
                    if x_component == y_component:
                        st.warning("Por favor, selecione componentes diferentes para X e Y.")
                    else:
                        # Visualização 2D com componentes selecionados
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(
                            plot_data[:, x_component], 
                            plot_data[:, y_component], 
                            c=st.session_state.clustering_labels, 
                            cmap='viridis', 
                            alpha=0.7
                        )
                        ax.set_title(f'Visualização 2D dos Clusters ({best_n_clusters_retrain} clusters)')
                        ax.set_xlabel(f'Componente {x_component+1}')
                        ax.set_ylabel(f'Componente {y_component+1}')
                        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                        ax.add_artist(legend)
                        st.pyplot(fig)
                        plt.clf()

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
                st.write("### Re-Treino do Modelo")
                
                # Escolha do método para determinar o número de clusters
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
                    if silhouette_scores and any(score > 0 for score in silhouette_scores):
                        best_n_clusters_retrain = range(num_clusters_range[0], num_clusters_range[1] + 1)[np.argmax(silhouette_scores)]
                    else:
                        st.error("Não foi possível determinar automaticamente o número de clusters. Por favor, selecione manualmente.")
                        best_n_clusters_retrain = 3  # Valor padrão
                        
                # Botão para executar o re-treino
                if st.button("Treinar Novamente"):
                    model = st.session_state.models[st.session_state.selected_model_name]
                    
                    # Preparar modelo
                    if st.session_state.selected_model_name == "Clustering Hierárquico":
                        model.set_params(n_clusters=best_n_clusters_retrain, linkage='ward')
                    else:
                        model.set_params(n_clusters=best_n_clusters_retrain, n_init=5, max_iter=300)
                    
                    # Treinar o modelo com uma barra de progresso
                    with st.spinner(f"Realizando re-treino com {best_n_clusters_retrain} clusters..."):
                        model.fit(st.session_state.training_data)
                    
                    # Calcular métricas
                    st.session_state.retrain_metrics = {
                        "Número de Clusters": best_n_clusters_retrain,
                        "Silhouette Score": silhouette_score(st.session_state.training_data, model.labels_),
                        "Davies-Bouldin Index": davies_bouldin_score(st.session_state.training_data, model.labels_),
                        "Calinski-Harabasz Score": calinski_harabasz_score(st.session_state.training_data, model.labels_)
                    }
                    
                    # Atualizar rótulos dos clusters
                    st.session_state.retrain_labels = model.labels_
                    st.session_state.retrain_completed = True
                    
                    # Mensagem de sucesso
                    if st.session_state.selected_model_name == "Clustering Hierárquico":
                        st.success(f"Re-treino concluído com sucesso com {best_n_clusters_retrain} clusters e {st.session_state.pca_n_components} componentes PCA!")
                    else:
                        st.success(f"Re-treino concluído com sucesso com {best_n_clusters_retrain} clusters!")
                    
                # Exibir métricas do re-treino após a execução
                if st.session_state.get("retrain_completed", False):
                    st.write("### Métricas do Re-Treino")
                    st.table(fix_dataframe_types(pd.DataFrame([st.session_state.retrain_metrics])))
                    
                    # Recuperar o modelo do estado da sessão
                    current_model = st.session_state.models[st.session_state.selected_model_name]

                    # Verificar centroides para KMeans
                    if st.session_state.selected_model_name == "KMeans":
                        if hasattr(current_model, 'cluster_centers_'):
                            st.write("#### Centroides dos Clusters")
                            centroids = current_model.cluster_centers_
                            if centroids.shape[1] > 10:
                                st.write(f"(Mostrando apenas as primeiras 10 dimensões de {centroids.shape[1]})")
                                centroids_df = pd.DataFrame(centroids[:, :10])
                            else:
                                centroids_df = pd.DataFrame(centroids)
                            
                            st.dataframe(fix_dataframe_types(centroids_df))
    
                    # Visualização dos clusters do re-treino
                    if 'retrain_labels' in st.session_state:
                        st.write("### Visualização dos Clusters do Re-Treino")
                        
                        # Preparar dados para visualização
                        if st.session_state.selected_model_name == "Clustering Hierárquico":
                            # Para hierárquico, já temos os dados PCA
                            plot_data = st.session_state.X_pca
                        else:
                            # Para K-means, aplicamos um novo PCA para visualização
                            # Use os dados originais ou X_scaled
                            X_for_viz = X_scaled  # ou outro conjunto de dados apropriado
                            if X_for_viz.shape[1] > 3:
                                pca_viz = PCA(n_components=3)
                                plot_data = pca_viz.fit_transform(X_for_viz)
                                st.write("(Dados reduzidos via PCA para visualização)")
                            else:
                                plot_data = X_for_viz
                        
                        # Obter número total de componentes
                        total_components = plot_data.shape[1]
                        
                        # Permitir escolha de componentes para x e y
                        st.write("### Escolha os Componentes para Visualização")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_component = st.selectbox(
                                "Componente para o Eixo X", 
                                list(range(total_components)), 
                                index=0,
                                format_func=lambda x: f"Componente {x+1}",
                                key="retrain_x_component"  # Chave única adicionada
                            )
                        
                        with col2:
                            y_component = st.selectbox(
                                "Componente para o Eixo Y", 
                                list(range(total_components)), 
                                index=1 if total_components > 1 else 0,
                                format_func=lambda x: f"Componente {x+1}",
                                key="retrain_y_component"  # Chave única adicionada
                            )
                        
                        # Verificar se componentes são diferentes
                        if x_component == y_component:
                            st.warning("Por favor, selecione componentes diferentes para X e Y.")
                        else:
                            # Visualização 2D com componentes selecionados
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(
                                plot_data[:, x_component], 
                                plot_data[:, y_component], 
                                c=st.session_state.retrain_labels, 
                                cmap='viridis', 
                                alpha=0.7
                            )
                            ax.set_title(f'Visualização 2D dos Clusters do Re-Treino ({best_n_clusters_retrain} clusters)')
                            ax.set_xlabel(f'Componente {x_component+1}')
                            ax.set_ylabel(f'Componente {y_component+1}')
                            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                            ax.add_artist(legend)
                            st.pyplot(fig)
                            plt.clf()
                            
                # Finalizar após o re-treino
                if st.session_state.get("retrain_completed", False):
                    st.write("## Concluir o Processo de Clustering")
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
                # Perguntar ao utilizador se quer usar GridSearch
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
                    
                        # Se não houver parâmetros padrão, informar o utilizador
                        if not param_grid:
                            st.warning(f"Parâmetros padrão não definidos para o modelo {model_key}.")
                            param_grid = {}
                    
                        # Exibir os parâmetros para o utilizador ajustar manualmente
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
                st.write(f"GridSearch Ativado? {use_grid_search}")  # Debug para verificar a escolha do utilizador

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
    try:
        # Padronizar os dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        if model_name == "KMeans":
            model.set_params(n_clusters=st.session_state.kmeans_clusters)
            model.fit(X_scaled)
            st.session_state['labels'] = model.labels_
        
        elif model_name == "Clustering Hierárquico":
            # Configurar explicitamente todos os parâmetros necessários
            model.set_params(n_clusters=st.session_state.kmeans_clusters, linkage="ward")
            model.fit(X_scaled)
            st.session_state['labels'] = model.labels_
        
        st.write(f"Clusterização realizada com {model_name}")
        
    except Exception as e:
        st.error(f"Erro ao treinar o modelo {model_name}: {str(e)}")
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
    try:
        # Verificações para tipos de modelos
        is_svr = isinstance(model, SVR)
        is_svc = isinstance(model, SVC)  # Adicionar verificação para SVC
        is_regression = is_svr or isinstance(model, LinearRegression)

        # Escalonamento para SVR
        if is_svr:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # GridSearch otimizado
        if use_grid_search:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2' if is_regression else 'accuracy'
            
            # Tratamento especial para SVC (muito mais rápido)
            if is_svc:
                # Grid reduzido para SVC
                simplified_grid = {
                    'C': [1],            # Apenas um valor para C
                    'kernel': ['rbf'],   # Apenas um tipo de kernel
                    'gamma': ['scale']   # Apenas uma configuração de gamma
                }
                
                # Aplicar parâmetros manuais, se fornecidos
                if manual_params:
                    for param, value in manual_params.items():
                        simplified_grid[param] = [value]
                        
                # Usar o grid simplificado
                actual_grid = simplified_grid
                
                # Reduzir número de folds para SVC
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
            else:
                # Para outros modelos, usar o grid original
                actual_grid = param_grid
                
                # Incorporar parâmetros manuais
                if manual_params:
                    actual_grid.update({k: [v] for k, v in manual_params.items()})
            
            # Executar GridSearch com os parâmetros apropriados
            grid_search = GridSearchCV(
                model, 
                actual_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1  # Utilizar todos os cores disponíveis
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Treinar sem GridSearch
            if manual_params:
                model.set_params(**manual_params)
            
            model.fit(X_train, y_train)
            best_model = model
            best_params = manual_params or {}

        # Predições
        y_pred = best_model.predict(X_test)

        # Métricas baseadas no tipo de modelo
        metrics = {
            "Modelo": model.__class__.__name__,
            **(
                {
                    "R²": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred)
                } if is_regression else 
                {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1-Score": f1_score(y_test, y_pred, average='weighted')
                }
            ),
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
    
    if 'feature_selection_done' not in st.session_state:
        st.session_state.feature_selection_done = False
    
    model_type = st.session_state.get('model_type', 'Classificação')
    scoring_options = {"Classificação": ['Accuracy', 'Precision', 'Recall', 'F1-Score'], "Regressão": ['R²', 'MAE', 'MSE']}
    
    selected_scoring = st.selectbox("Escolha a métrica de scoring:", scoring_options.get(model_type, []))
    
    if st.button("Confirmar Scoring"):
        st.session_state.selected_scoring = selected_scoring
        st.session_state.scoring_confirmed = True
        st.success(f"Métrica de scoring {selected_scoring} confirmada!")
    
    if st.session_state.scoring_confirmed:
        method_selection = st.radio("Escolha o método de seleção de features:", ["Automático", "Manual"])
        
        if st.button("Confirmar Método"):
            st.session_state.method_selection = method_selection
            st.success(f"Método {method_selection} confirmado!")

        X_train, X_test, y_train, y_test = st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test
        
        if method_selection == "Automático":
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42) if model_type == "Classificação" else RandomForestRegressor(n_estimators=100, random_state=42)
            feature_selector.fit(X_train, y_train)
            
            feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': feature_selector.feature_importances_}).sort_values('importance', ascending=False)
            st.dataframe(feature_importances)
            
            selected_features = feature_importances[feature_importances['importance'] > 0.01]['feature'].tolist()
        else:
            feature_selector = RandomForestClassifier(n_estimators=100, random_state=42) if model_type == "Classificação" else RandomForestRegressor(n_estimators=100, random_state=42)
            feature_selector.fit(X_train, y_train)
            
            feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': feature_selector.feature_importances_}).sort_values('importance', ascending=False)
            st.dataframe(feature_importances)
            
            num_features = st.slider("Número de Features a Selecionar", 1, X_train.shape[1], min(5, X_train.shape[1]))
            selected_features = feature_importances['feature'].head(num_features).tolist()

        st.session_state.X_train_selected = X_train[selected_features]
        st.session_state.X_test_selected = X_test[selected_features]
        st.session_state.selected_features = selected_features
        st.session_state.feature_selection_done = True

        if st.button("Treinar Modelo com Features Selecionadas"):
            st.session_state.step = 'train_with_selected_features'
            st.rerun()

def train_with_selected_features_page():
    st.title("Treino do Modelo com Features Selecionadas")
    
    # Mapeamento de modelos bidirecional
    model_name_map = {
        "SVC": "Support Vector Classification (SVC)",
        "KNeighborsClassifier": "K-Nearest Neighbors (KNN)",
        "RandomForestClassifier": "Random Forest",
        "LinearRegression": "Regressão Linear Simples (RLS)",
        "SVR": "Regressão por Vetores de Suporte (SVR)",
        "Support Vector Classification (SVC)": "SVC",
        "K-Nearest Neighbors (KNN)": "KNeighborsClassifier", 
        "Random Forest": "RandomForestClassifier",
        "Regressão Linear Simples (RLS)": "LinearRegression",
        "Regressão por Vetores de Suporte (SVR)": "SVR"
    }
    
    if 'models' not in st.session_state or not st.session_state.models:
        st.error("Erro: Nenhum modelo foi treinado ou selecionado.")
        return

    if 'selected_model_name' not in st.session_state or not st.session_state.selected_model_name:
        st.error("Nenhum modelo foi selecionado. Por favor, selecione um modelo antes de continuar.")
        return

    selected_model_name = st.session_state.selected_model_name.strip()
    model_class_name = model_name_map.get(selected_model_name, selected_model_name)

    if model_class_name not in st.session_state.models:
        st.error(f"O modelo '{selected_model_name}' não foi encontrado na sessão.")
        st.write("Modelos disponíveis:", list(st.session_state.models.keys()))
        return

    model = st.session_state.models[model_class_name]
    
    X_train_selected, X_test_selected = st.session_state.X_train_selected, st.session_state.X_test_selected
    y_train, y_test = st.session_state.y_train, st.session_state.y_test
    
    st.write(f"Treinando o modelo {selected_model_name} com {len(st.session_state.selected_features)} features selecionadas...")
    
    selected_metrics = train_and_store_metrics(model, X_train_selected, y_train, X_test_selected, y_test, "Com Seleção", False)
    
    if selected_metrics:
        st.session_state['resultado_com_selecao'] = selected_metrics
        st.success("Treinamento concluído!")
        
        st.subheader("Métricas do Modelo com Features Selecionadas")
        metrics_df = pd.DataFrame([selected_metrics])
        metrics_df.insert(0, "Modelo", "Com Seleção de Features")
        st.table(metrics_df)
    
    if st.button("Comparar Modelos"):
        st.session_state.step = 'evaluate_and_compare_models'
        st.rerun()

#Função para Treinar e Armazenar as metricas

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

def evaluate_and_compare_models():
    st.title("Comparação dos Resultados do Treino dos Modelos")

    # Mapeamento de modelos bidirecional
    model_name_map = {
        "SVC": "Support Vector Classification (SVC)",
        "KNeighborsClassifier": "K-Nearest Neighbors (KNN)",
        "RandomForestClassifier": "Random Forest",
        "LinearRegression": "Regressão Linear Simples (RLS)",
        "SVR": "Regressão por Vetores de Suporte (SVR)",
        "Support Vector Classification (SVC)": "SVC",
        "K-Nearest Neighbors (KNN)": "KNeighborsClassifier", 
        "Random Forest": "RandomForestClassifier",
        "Regressão Linear Simples (RLS)": "LinearRegression",
        "Regressão por Vetores de Suporte (SVR)": "SVR"
    }

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

    # Recuperar a métrica escolhida pelo utilizador para seleção de features
    scoring_metric = st.session_state.get("selected_scoring", None)
    if not scoring_metric:
        st.error("Nenhuma métrica de avaliação foi escolhida. Por favor, volte à etapa de seleção de métricas.")
        return

    # Recuperar o nome do modelo selecionado
    model_name = st.session_state.get('selected_model_name')
    if not model_name:
        st.error("Nenhum modelo foi selecionado. Por favor, volte à etapa de seleção de modelos.")
        return

    # Encontrar o nome correto do modelo a partir do mapeamento
    model_class_name = model_name_map.get(model_name)
    if model_class_name is None:
        st.error(f"O modelo {model_name} não foi encontrado na lista de modelos disponíveis.")
        st.write("Modelos disponíveis:", list(model_name_map.keys()))
        return

    # Recuperar o modelo da sessão com base no nome correto da classe
    model = st.session_state.models.get(model_class_name)
    if model is None:
        st.error(f"O modelo {model_class_name} não foi encontrado na sessão.")
        st.write("Modelos disponíveis:", list(st.session_state.models.keys()))
        return

    # Recuperar métricas originais e com seleção de features
    original_metrics = st.session_state.get('resultado_sem_selecao', {}) 
    selected_metrics = st.session_state.get('resultado_com_selecao', {})

    # Verificar se as métricas existem
    if not original_metrics:
        st.error("Não foi possível encontrar as métricas originais. Por favor, refaça o treinamento.")
        return
        
    if not selected_metrics:
        st.error("Não foi possível encontrar as métricas com seleção de features. Por favor, execute o treino com features selecionadas.")
        return

    # Criar DataFrame de comparação
    if model_type == "Classificação":
        comparison_df = pd.DataFrame({
            'Modelo': ['Sem Seleção de Features', 'Com Seleção de Features'],
            'Accuracy': [original_metrics.get('Accuracy', 0), selected_metrics.get('Accuracy', 0)],
            'Precision': [original_metrics.get('Precision', 0), selected_metrics.get('Precision', 0)],
            'Recall': [original_metrics.get('Recall', 0), selected_metrics.get('Recall', 0)],
            'F1-Score': [original_metrics.get('F1-Score', 0), selected_metrics.get('F1-Score', 0)],
            'Best Parameters': [original_metrics.get('Best Parameters', 'N/A'), selected_metrics.get('Best Parameters', 'N/A')]
        })
    elif model_type == "Regressão":
        comparison_df = pd.DataFrame({
            'Modelo': ['Sem Seleção de Features', 'Com Seleção de Features'],
            'R²': [original_metrics.get('R²', 0), selected_metrics.get('R²', 0)],
            'MAE': [original_metrics.get('MAE', 0), selected_metrics.get('MAE', 0)],
            'MSE': [original_metrics.get('MSE', 0), selected_metrics.get('MSE', 0)],
            'Best Parameters': [original_metrics.get('Best Parameters', 'N/A'), selected_metrics.get('Best Parameters', 'N/A')]
        })
    else:
        st.error(f"Tipo de modelo não reconhecido: {model_type}")
        return

    # Exibir tabela de comparação
    st.subheader("📈 Comparação dos Resultados:")
    
    # Formatar todas as colunas numéricas
    format_dict = {}
    for col in comparison_df.columns:
        if col != 'Modelo' and col != 'Best Parameters':
            format_dict[col] = '{:.4f}'
    
    st.dataframe(
    comparison_df.style.format(format_dict).set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '18px')]}, 
         {'selector': 'td', 'props': [('font-size', '12px')]},  
         {'selector': 'table', 'props': [('width', '100%')]},    
        ]
    )
)
    
    # Determinar as métricas disponíveis com base no tipo de modelo
    if model_type == "Classificação":
        metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    elif model_type == "Regressão":
        metric_columns = ['R²', 'MAE', 'MSE']
    else:
        metric_columns = []
    
    # Garantir que a métrica escolhida existe nas colunas disponíveis
    if scoring_metric not in metric_columns:
        st.warning(f"A métrica selecionada '{scoring_metric}' não está disponível. Usando a primeira métrica disponível.")
        scoring_metric = metric_columns[0] if metric_columns else None
    
    if scoring_metric:
        # Gráfico de comparação usando a métrica escolhida pelo utilizador

        x = comparison_df['Modelo']
        y1 = comparison_df[scoring_metric].iloc[0]  # Sem Seleção de Features (índice 0)
        y2 = comparison_df[scoring_metric].iloc[1]  # Com Seleção de Features (índice 1)

        # Gráfico de comparação com melhorias no layout e visibilidade dos rótulos
        fig, ax = plt.subplots(figsize=(10, 6))

        # Posições das barras
        x_pos = [0, 1]  # Definindo a posição das barras para garantir que fiquem ao lado
        width = 0.4  # Largura das barras

        # Ajustar as barras para uma boa visibilidade
        bars1 = ax.bar(x_pos[0], y1, width=width, label="Sem Seleção de Features", color='#90EE90', align='center')
        bars2 = ax.bar(x_pos[1], y2, width=width, label="Com Seleção de Features", color='#006400', align='center')

        # Adicionar rótulos de valor nas barras com melhorias
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, color='black')

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, color='black')  # Altere a cor para garantir contraste

        # Melhorando o título e as labels
        ax.set_title(f'Comparação de {scoring_metric}', fontsize=16, fontweight='bold')
        ax.set_ylabel(scoring_metric, fontsize=14)
        ax.set_xlabel("Modelos", fontsize=14)

        # Ajuste nos rótulos do eixo X e Y
        plt.xticks(x_pos, ['Sem Seleção de Features', 'Com Seleção de Features'], fontsize=12)
        plt.yticks(fontsize=12)

        # Legenda
        ax.legend()

        # Ajuste do layout para garantir que tudo fique visível
        plt.tight_layout()

        # Exibir o gráfico
        st.pyplot(fig)


    # Determinar o melhor modelo baseado na métrica escolhida
    if scoring_metric:
        score_without = comparison_df[scoring_metric].iloc[0]
        score_with = comparison_df[scoring_metric].iloc[1]
        
        better_model = "Com Seleção de Features" if score_with > score_without else "Sem Seleção de Features"
        better_score = max(score_with, score_without)
        
        st.success(f"🏆 **Melhor modelo:** {better_model} com {scoring_metric} = {better_score:.4f}")
    
    # Botão para próxima etapa
    if st.button("Seguir para Resumo Final", key="btn_resumo_final"):
        st.session_state.step = 'final_page'
        st.rerun()

# Função para gerar interpretação personalizada das métricas
def generate_metrics_interpretation(metrics):
    """Função para gerar interpretação personalizada das métricas"""
    interpretacao = []

    # Verificar se as métricas estão no formato esperado
    if not isinstance(metrics, dict):
        return "Formato de métricas inválido."

    # Accuracy
    if 'Accuracy' in metrics:
        try:
            accuracy = float(metrics['Accuracy'])
            if accuracy > 0.9:
                interpretacao.append(f"- Acurácia: {accuracy:.4f} - Excelente! O modelo tem uma taxa de acerto global muito elevada.")
            elif accuracy > 0.75:
                interpretacao.append(f"- Acurácia: {accuracy:.4f} - Boa. O modelo está a funcionar bem, mas ainda há margem para otimização.")
            elif accuracy > 0.5:
                interpretacao.append(f"- Acurácia: {accuracy:.4f} - Moderada. Os erros ainda são significativos e devem ser corrigidos.")
            else:
                interpretacao.append(f"- Acurácia: {accuracy:.4f} - Fraca. O modelo está a falhar em muitas previsões e precisa de ser revisto.")
        except (ValueError, TypeError):
            interpretacao.append("- Acurácia: Não disponível ou inválida.")

    # Precision
    if 'Precision' in metrics:
        try:
            precision = float(metrics['Precision'])
            if precision > 0.9:
                interpretacao.append(f"- Precisão: {precision:.4f} - Excelente! O modelo está a evitar a maioria dos falsos positivos.")
            elif precision > 0.75:
                interpretacao.append(f"- Precisão: {precision:.4f} - Bom. O modelo evita falsos positivos, mas pode ser mais rigoroso.")
            elif precision > 0.5:
                interpretacao.append(f"- Precisão: {precision:.4f} - Moderada. Há um número considerável de falsos positivos a corrigir.")
            else:
                interpretacao.append(f"- Precisão: {precision:.4f} - Fraca. Muitos falsos positivos estão a prejudicar a confiança nas previsões.")
        except (ValueError, TypeError):
            interpretacao.append("- Precisão: Não disponível ou inválida.")

    # Recall
    if 'Recall' in metrics:
        try:
            recall = float(metrics['Recall'])
            if recall > 0.9:
                interpretacao.append(f"- Recall: {recall:.4f} - Excelente! O modelo está a identificar quase todos os positivos verdadeiros.")
            elif recall > 0.75:
                interpretacao.append(f"- Recall: {recall:.4f} - Bom. A maioria dos positivos verdadeiros é identificada, mas há espaço para melhorias.")
            elif recall > 0.5:
                interpretacao.append(f"- Recall: {recall:.4f} - Moderado. O modelo está a perder demasiados positivos verdadeiros.")
            else:
                interpretacao.append(f"- Recall: {recall:.4f} - Fraco. O modelo falha em identificar a maioria dos positivos verdadeiros. Pode ser necessário ajustar os pesos ou thresholds.")
        except (ValueError, TypeError):
            interpretacao.append("- Recall: Não disponível ou inválido.")
    
    # F1-Score
    if 'F1-Score' in metrics:
        try:
            f1_score = float(metrics['F1-Score'])
            if f1_score > 0.9:
                interpretacao.append(f"- F1-Score: {f1_score:.4f} - Excelente equilíbrio entre precisão e sensibilidade. O modelo está altamente otimizado.")
            elif f1_score > 0.75:
                interpretacao.append(f"- F1-Score: {f1_score:.4f} - Bom desempenho. Contudo, há espaço para melhorias nos falsos positivos ou negativos.")
            elif f1_score > 0.5:
                interpretacao.append(f"- F1-Score: {f1_score:.4f} - Desempenho moderado. Ajustes no treino ou balanceamento dos dados podem ajudar.")
            else:
                interpretacao.append(f"- F1-Score: {f1_score:.4f} - Desempenho fraco. Recomenda-se rever os dados, ajustar hiperparâmetros ou otimizar o modelo.")
        except (ValueError, TypeError):
            interpretacao.append("- F1-Score: Não disponível ou inválido.")

    # Se nenhuma métrica conhecida foi encontrada
    if not interpretacao:
        interpretacao.append("Nenhuma métrica de classificação reconhecida encontrada nos dados.")

    # Conclusão Geral
    if all(key in metrics for key in ['F1-Score', 'Precision', 'Recall']):
        try:
            f1_score = float(metrics['F1-Score'])
            precision = float(metrics['Precision'])
            recall = float(metrics['Recall'])
            
            if f1_score > 0.9 and precision > 0.9 and recall > 0.9:
                interpretacao.append("\nConclusão Geral: 🎉 O modelo apresenta um desempenho excecional em todas as métricas. Está pronto para produção!")
            elif f1_score > 0.75 and precision > 0.75 and recall > 0.75:
                interpretacao.append("\nConclusão Geral: 👍 O modelo tem um bom desempenho geral, mas pode ser ligeiramente melhorado com ajustes finos.")
            elif f1_score > 0.5 or precision > 0.5 or recall > 0.5:
                interpretacao.append("\nConclusão Geral:⚠️ O modelo tem um desempenho moderado. Recomenda-se ajustar os hiperparâmetros ou melhorar os dados de treino.")
            else:
                interpretacao.append("\nConclusão Geral: ❗ O modelo apresenta um desempenho fraco. Será necessário rever o processo de treino, os dados e os parâmetros.")
        except (ValueError, TypeError):
            pass

    return "\n".join(interpretacao)

def generate_regression_interpretation(metrics):
    """Função para gerar interpretação personalizada das métricas de regressão"""
    interpretation = []

    # Verificar se as métricas estão no formato esperado
    if not isinstance(metrics, dict):
        return "Formato de métricas inválido."

    # R² (Coeficiente de Determinação)
    if 'R²' in metrics:
        try:
            r2 = float(metrics['R²'])
            if r2 > 0.9:
                interpretation.append(f"- R²: {r2:.4f} - Excelente! O modelo explica quase toda a variabilidade dos dados. Isso indica um forte ajuste entre as previsões e os valores reais.")
            elif r2 > 0.75:
                interpretation.append(f"- R²: {r2:.4f} - Muito bom! O modelo explica a maior parte da variabilidade dos dados, mas ainda pode ser melhorado.")
            elif r2 > 0.5:
                interpretation.append(f"- R²: {r2:.4f} - Moderado. O modelo consegue explicar uma parte significativa da variabilidade, mas há limitações importantes no ajuste.")
            else:
                interpretation.append(f"- R²: {r2:.4f} - Fraco. O modelo explica pouca variabilidade dos dados. Considere revisar as features ou usar um modelo mais adequado.")
        except (ValueError, TypeError):
            interpretation.append("- R²: Não disponível ou inválido.")

    # MAE (Erro Absoluto Médio)
    if 'MAE' in metrics:
        try:
            mae = float(metrics['MAE'])
            if mae < 0.1:
                interpretation.append(f"- MAE: {mae:.4f} - Excelente! O erro absoluto médio é muito pequeno, sugerindo que as previsões são altamente precisas.")
            elif mae < 1:
                interpretation.append(f"- MAE: {mae:.4f} - Bom. O erro absoluto médio é aceitável, mas ainda pode ser otimizado.")
            else:
                interpretation.append(f"- MAE: {mae:.4f} - Alto. As previsões estão frequentemente desviando dos valores reais. Considere ajustar o modelo ou as features.")
        except (ValueError, TypeError):
            interpretation.append("- MAE: Não disponível ou inválido.")

    # MSE (Erro Quadrático Médio)
    if 'MSE' in metrics:
        try:
            mse = float(metrics['MSE'])
            if mse < 0.1:
                interpretation.append(f"- MSE: {mse:.4f} - Excelente! O erro quadrático médio é muito baixo, indicando que as previsões estão próximas dos valores reais.")
            elif mse < 1:
                interpretation.append(f"- MSE: {mse:.4f} - Bom. O erro é relativamente baixo, mas ainda há espaço para reduzir as discrepâncias.")
            else:
                interpretation.append(f"- MSE: {mse:.4f} - Alto. O erro é significativo. Isso pode indicar que o modelo não está capturando bem os padrões nos dados.")
        except (ValueError, TypeError):
            interpretation.append("- MSE: Não disponível ou inválido.")

    # Se nenhuma métrica conhecida foi encontrada
    if not interpretation:
        interpretation.append("Nenhuma métrica de regressão reconhecida encontrada nos dados.")

    # Conclusão geral com base nas métricas
    if all(key in metrics for key in ['R²', 'MAE', 'MSE']):
        try:
            r2 = float(metrics['R²'])
            mse = float(metrics['MSE'])
            mae = float(metrics['MAE'])
            
            if r2 > 0.9 and mse < 0.1 and mae < 0.1:
                interpretation.append("\nConclusão Geral: 🎉 O modelo apresenta um desempenho excepcional! Está pronto para produção.")
            elif r2 > 0.75 and mse < 1 and mae < 1:
                interpretation.append("\nConclusão Geral: 👍 O modelo tem um bom desempenho geral. Com ajustes menores, pode se tornar ainda melhor.")
            elif r2 > 0.5 or mse < 1 or mae < 1:
                interpretation.append("\nConclusão Geral: ⚠️ O modelo está funcional, mas ainda apresenta limitações. Ajustes adicionais são recomendados.")
            else:
                interpretation.append("\nConclusão Geral: ❗ O modelo apresenta desempenho insatisfatório. Considere reavaliar as features, ajustar hiperparâmetros ou explorar modelos alternativos.")
        except (ValueError, TypeError):
            pass

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
from fpdf import FPDF
import requests
import tempfile
from datetime import datetime
from io import BytesIO

class CustomPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Baixar o logo no início para reutilizá-lo
        self.logo_path = None
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        try:
            response = requests.get(logo_url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    tmpfile.write(response.content)
                    self.logo_path = tmpfile.name
        except Exception as e:
            print(f"Erro ao baixar o logo: {e}")

    def header(self):
        # Posicionar o cabeçalho no topo da página
        self.set_y(10)
        
        # Adicionar a imagem no cabeçalho se o logo foi baixado com sucesso
        if self.logo_path:
            self.image(self.logo_path, 10, 10, 25)
        
        # Configurar fonte para o título
        self.set_font('Arial', 'B', 12)
        
        # Adicionar o título centralizado
        # Deixar espaço para o logo
        self.cell(25)  # Espaço para o logo
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', 0, 0, 'C')
        
        # Adicionar uma linha horizontal após o cabeçalho
        self.ln(15)
        self.ln(5)  # Espaço após o cabeçalho

    def footer(self):
        # Ir para 1.5 cm da parte inferior
        self.set_y(-20)
        
        # Adicionar uma linha horizontal antes do rodapé
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        
        # Definir fonte para o rodapé
        self.set_font('Arial', 'I', 8)
        
        # Data atual
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # Adicionar rodapé com a data e número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', 0, 0, 'C')
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

    # Inicialização do PDF com cabeçalho e rodapé
    pdf = CustomPDF(format='A4')
    pdf.set_margins(10, 30, 10)  # left, top, right
    pdf.set_auto_page_break(auto=True, margin=30)  # Margem inferior para o rodapé
    pdf.add_page()
    
    # Função para limpar texto para compatibilidade com codificação Latin-1
    def clean_text(text):
        if not isinstance(text, str):
            return str(text)
        return text.encode('latin-1', errors='ignore').decode('latin-1')
    

    # Título do Relatório
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, txt=clean_text("Relatório Final do Modelo Treinado"), ln=True, align="C")
    pdf.ln(10)
    
    # Tipo de Modelo
    model_type = session_state.get('model_type', 'Indefinido')
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Tipo de Modelo:"), ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=clean_text(model_type), ln=True)
    
    # Modelo Selecionado
    selected_model_name = session_state.get('selected_model_name', 'Não Selecionado')
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Modelo Selecionado:"), ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=clean_text(selected_model_name), ln=True)
    
    # Melhor Modelo
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(60, 10, txt=clean_text("Melhor Modelo:"), ln=False)
    pdf.set_font("Arial", size=12)
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
        
        pdf.set_font("Arial", style="B", size=14)
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
        pdf.set_font("Arial", size=10)
        pdf.set_fill_color(144, 238, 144) # Cor de fundo do cabeçalho
        
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
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, txt=clean_text("Features Selecionadas"), ln=True)
        
        # Listar as features
        features = session_state.selected_features
        pdf.set_font("Arial", size=10)
        for i, feature in enumerate(features):
            pdf.cell(0, 6, txt=clean_text(f"• {feature}"), ln=True)
        
        pdf.ln(10)
    
    # Tabela de Métricas
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Comparação de Métricas"), ln=True)
    
    # Verificar o tipo de modelo para determinar quais métricas exibir
    is_regression = model_type == "Regressão"
    metric_columns = ['R²', 'MAE', 'MSE'] if is_regression else ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Criar tabela de métricas
    pdf.set_font("Arial", style="B", size=10)
    pdf.set_fill_color(144, 238, 144) # Cor de fundo do cabeçalho
    
    # Definir a largura das colunas
    column_width = 30
    first_column_width = 60
    
    # Cabeçalho da tabela
    pdf.cell(first_column_width, 10, "Modelo", 1, 0, 'C', True)
    for col in metric_columns:
        pdf.cell(column_width, 10, clean_text(col), 1, 0, 'C', True)
    pdf.ln()
    
    # Linhas da tabela
    pdf.set_font("Arial", size=10)
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
    for metric in metric_columns:
        if metric in comparison_df.columns:
            # Criar o gráfico com tamanho ajustado
            plt.figure(figsize=(10, 6))
            
            # Dados para o gráfico
            models = comparison_df['Modelo'].tolist()
            values = comparison_df[metric].tolist()
            
            # Criar barras com espaçamento adequado
            plt.bar(models, values, color=['#90EE90', '#006400'], width=0.4)
            
            # Adicionar valores sobre as barras
            for i, v in enumerate(values):
                if isinstance(v, (int, float)):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
            
            # MUDANÇA PRINCIPAL: Configuração do eixo X sem rotação
            plt.xticks(rotation=0, ha='center', fontsize=8)  # Mudar rotation=45 para rotation=0
            
            # Estilização com mais espaço
            plt.title(f"Comparação de {metric}", fontsize=14, pad=15)  # Aumentar pad para dar mais espaço
            plt.ylabel(metric, fontsize=12)
            
            # Garantir espaço para o conteúdo
            plt.subplots_adjust(bottom=0.2, left=0.15)  # Aumentar margem inferior
            
            # Ajustar a altura do gráfico para evitar corte
            plt.ylim(0, max(values) * 1.2)  # Aumenta o limite superior em 20%
            
            plt.tight_layout()  # Ajusta automaticamente o layout
            
            # Salvar o gráfico em um arquivo temporário com DPI maior
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)  # Aumentar DPI e garantir que nada seja cortado
            plt.close()
        
            # Adicionar o gráfico ao PDF - AJUSTADO
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(0, 10, txt=clean_text(f"Gráfico de Comparação - {metric}"), ln=True, align="C")
            
            # Posicionar o gráfico mais para baixo para evitar sobreposição com o cabeçalho
            pdf.image(temp_file.name, x=10, y=45, w=180)  # Posição Y aumentada
            
            # Fechar e remover o arquivo temporário
            temp_file.close()
            try:
                os.remove(temp_file.name)
            except:
                pass  # Ignorar erros ao remover arquivos temporários
        
    # Interpretação das Métricas
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Interpretação das Métricas"), ln=True, align="C")
    
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
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Modelo Sem Seleção de Features"), ln=True)
    pdf.set_font("Arial", size=10)
    
    # Adicionar interpretação do modelo sem seleção
    for line in generate_metrics_interpretation(original_metrics, model_type):
        pdf.multi_cell(0, 8, txt=clean_text(f"• {line}"))
    
    pdf.ln(5)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt=clean_text("Modelo Com Seleção de Features"), ln=True)
    pdf.set_font("Arial", size=10)
    
    # Adicionar interpretação do modelo com seleção
    for line in generate_metrics_interpretation(selected_metrics, model_type):
        pdf.multi_cell(0, 8, txt=clean_text(f"• {line}"))
    
    # Conclusão
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt=clean_text("Conclusão"), ln=True)
    
    # Determinar a melhor métrica com base na escolha do utilizador
    scoring_metric = session_state.get("selected_scoring", None)

    # Fallback para métricas padrão se a métrica selecionada não estiver disponível
    if not scoring_metric or scoring_metric not in metric_columns:
        main_metric = 'R²' if is_regression else 'F1-Score'
    else:
        main_metric = scoring_metric

    # Obter os valores da métrica escolhida
    original_value = original_metrics.get(main_metric, 0)
    selected_value = selected_metrics.get(main_metric, 0)

    # Texto da conclusão
    pdf.set_font("Arial", size=10)
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
    import unidecode # Normaliza caracteres acentuados, útil para lidar com strings em diferentes idiomas.
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
    st.write(f"Métrica original: {metric}, limpa: {metric_clean}, mapeada: {mapped_metric}")
    
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

    # Exibir tabela de métricas com ajustes finos
    st.dataframe(comparison_df.style.format({
        'Accuracy': '{:.4f}' if 'Accuracy' in comparison_df.columns else None,
        'Precision': '{:.4f}' if 'Precision' in comparison_df.columns else None,
        'Recall': '{:.4f}' if 'Recall' in comparison_df.columns else None,
        'F1-Score': '{:.4f}' if 'F1-Score' in comparison_df.columns else None,
        'R²': '{:.4f}' if 'R²' in comparison_df.columns else None,
        'MAE': '{:.4f}' if 'MAE' in comparison_df.columns else None,
        'MSE': '{:.4f}' if 'MSE' in comparison_df.columns else None,
    }).set_table_styles([
        {'selector': 'th', 'props': [('font-size', '14px'), ('background-color', '#f0f0f0'), ('text-align', 'center'), ('font-weight', 'bold')]},  # Cabeçalho
        {'selector': 'td', 'props': [('font-size', '14px'), ('text-align', 'center')]},  # Tamanho das células e alinhamento
        {'selector': 'table', 'props': [('width', '100%'), ('border-collapse', 'collapse')]},  # Largura da tabela e bordas
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},  # Cor de fundo alternada para as linhas
        {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},  # Cor de fundo para linhas ímpares
    ]))

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
            comparison_df[selected_metric],
            color=['#9ACD32', '#006400']  # Verde claro e verde escuro
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
        if score_with_selection > score_without_selection:
            best_model = "Com Seleção de Features"
            best_score = score_with_selection
        else:
            best_model = "Sem Seleção de Features"
            best_score = score_without_selection
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
    model_filename = save_best_model(model, with_feature_selection=(best_model == "Com Seleção de Features"))

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
        pdf_file_name = f"relatorio_final_{st.session_state.get('selected_model_name', 'modelo')}.pdf"
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Baixar o logo no início para reutilizá-lo
        self.logo_path = None
        logo_url = 'https://www.ipleiria.pt/normasgraficas/wp-content/uploads/sites/80/2017/09/estg_v-01.jpg'
        try:
            response = requests.get(logo_url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    tmpfile.write(response.content)
                    self.logo_path = tmpfile.name
        except Exception as e:
            print(f"Erro ao baixar o logo: {e}")

    def header(self):
        # Posicionar o cabeçalho no topo da página
        self.set_y(10)
        
        # Adicionar a imagem no cabeçalho se o logo foi baixado com sucesso
        if self.logo_path:
            self.image(self.logo_path, 10, 10, 25)
        
        # Configurar fonte para o título
        self.set_font('Arial', 'B', 12)
        
        # Adicionar o título centralizado
        # Deixar espaço para o logo
        self.cell(25)  # Espaço para o logo
        self.cell(0, 10, 'MLCase - Plataforma de Machine Learning', 0, 0, 'C')
        
        # Adicionar uma linha horizontal após o cabeçalho
        self.ln(15)
        self.ln(5)  # Espaço após o cabeçalho

    def footer(self):
        # Ir para 1.5 cm da parte inferior
        self.set_y(-20)
        
        # Adicionar uma linha horizontal antes do rodapé
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        
        # Definir fonte para o rodapé
        self.set_font('Arial', 'I', 8)
        
        # Data atual
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # Adicionar rodapé com a data e número da página
        self.cell(0, 10, f'{current_date} - Página {self.page_no()}  |  Autora da Plataforma: Bruna Sousa', 0, 0, 'C')
# Função para gerar o relatório PDF
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

def gerar_relatorio_clustering_pdf(initial_metrics, retrain_metrics, best_model_type, st_session):
    pdf = CustomPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, txt="Relatório Final do Modelo Treinados", ln=True, align="C")
    pdf.ln(10)

    # Modelo Selecionado
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(50, 10, txt="Modelo Selecionado:", ln=False)
    pdf.set_font("Arial", size=12)
    model_info = st_session['selected_model_name']
    
    # Adicionar informação de componentes para KMeans e Clustering Hierárquico
    if st_session['selected_model_name'] in ["KMeans", "Clustering Hierárquico"]:
        model_info += f" (PCA: {st_session.get('pca_n_components', 'N/A')} componentes)"
    
    pdf.cell(0, 10, txt=model_info, ln=True)
    pdf.ln(5)

    # Melhor Modelo
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(50, 10, txt="Melhor Modelo Treinado:", ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=best_model_type, ln=True)
    pdf.ln(10)

    # Adicionar métricas do treino inicial e re-treino
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Métricas Obtidas", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.set_fill_color(200, 220, 255)

    # Cabeçalho da tabela
    pdf.set_fill_color(144, 238, 144)  # Verde claro (light green)
    col_widths = [40, 40, 40, 40]
    pdf.cell(col_widths[0], 10, "Treino", 1, 0, 'C', True)
    pdf.cell(col_widths[1], 10, "Silhouette Score", 1, 0, 'C', True)
    pdf.cell(col_widths[2], 10, "Davies-Bouldin Index", 1, 0, 'C', True)
    pdf.cell(col_widths[3], 10, "Calinski-Harabasz Score", 1, 1, 'C', True)

    # Dados da tabela
    pdf.set_font("Arial", size=8)
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
        pdf.set_font("Arial", size=10)
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
        plt.bar(labels, values, color=['#90EE90', '#006400'], edgecolor='black')
        plt.title(f"{metric} por Treino")
        plt.ylabel(metric)
        plt.xlabel("Treino")
        graph_path = f"temp_{metric.replace(' ', '_')}.png"
        plt.savefig(graph_path)
        plt.close()
        graphs.append(graph_path)

    # Inserir gráficos no PDF
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, txt="Gráficos das Métricas", ln=True, align='C')
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

    # Adicionar informação sobre o número de componentes
    if st.session_state.selected_model_name in ["KMeans", "Clustering Hierárquico"]:
        st.write(f"**Número de Componentes PCA:** {st.session_state.get('pca_n_components', 'N/A')}")
    
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
            "Clustering Hierárquico": AgglomerativeClustering(linkage='ward'),
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

    # Exibir estado atual para depuração
    #st.write(f"📌 Estado atual: {st.session_state.step}")

    # Roteamento baseado no estado atual
    if st.session_state.step == 'file_upload':
        upload_file()
    elif st.session_state.step == 'data_preview':
        data_preview()
    elif st.session_state.step == 'missing_values':
        handle_missing_values()
    elif st.session_state.step == 'outlier_detection':
        outlier_detection()
    elif st.session_state.step == 'data_summary':
        data_summary()
    elif st.session_state.step == 'model_selection':
        model_selection()
    elif st.session_state.step == 'feature_selection':
        feature_selection()
    elif st.session_state.step == 'train_with_selected_features':  
        train_with_selected_features_page()
    elif st.session_state.step == 'evaluate_and_compare_models':
        evaluate_and_compare_models()
    elif st.session_state.step == 'clustering_final_page':  # ✅ Adicionado!
        clustering_final_page()  # ✅ Chama a função do relatório final de clustering
    elif st.session_state.step == 'final_page':
        final_page()
    else:
        st.error(f"⚠ Etapa desconhecida: {st.session_state.step}. Reiniciando a aplicação.")
        st.session_state.step = 'file_upload'
        st.rerun()
        
    # Exibir o estado após a execução para depuração
    #st.write(f"Estado final: {st.session_state.step}")


if __name__ == "__main__":
    main()
