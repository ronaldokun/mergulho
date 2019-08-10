# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introducão
#
# A técnica de *Deep Learning* ( Redes Neurais Profundas ou simplesmente Aprendizagem Profunda ) está em alta, ouvimos o termo sendo mencionado em toda parte como o estado da arte da Aprendizagem de Máquina Moderna. No entanto ao vasculharmos o Youtube ou os tutoriais do Medium é ampla as diversas aplicações de *Deep Learning* em dados não estruturados, ou seja, que não possuem uma estrutura pré-definida, como fotos, vídeos e textos. As Redes Neurais Profundas se mostraram extremamente eficazes nesses domínios.
#
# No entanto, *Deep Learning* é igualmente poderoso para classificar dados estruturados como:
# * Bancos de Dados SQL
# * DataFrame ( Pandas ou R )
# * Planilhas de Dados, incluindo Séries Temporais ( e.g. Excel )
#
# Em particular na criação de **matrizes de peso** (*embeddings*) para as variáveis categóricas. Esses tipos de dados podem ser classificados de diversas maneiras, aqui iremos referi-los como *Dados Tabulares*. Eles são comumente utilizados na indústria, no entanto recebem bem menos atenção do que as aplicações em imagens, vídeos e textos. 
#
# A motivação para esta postagem, além do cunho pessoal de se tratar de notas de aula, é devido a relevância desse modelo para aplicações práticas.

# %% [markdown]
# ## Apresentação do Problema
# Para ilustrar tal aplicação, efetuaremos uma solução de ponta a ponta para o Desafio do Kaggle: [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales). A tradução livre da apresentação do desafio é a seguinte:

# %% [markdown]
# A rede Rossman opera mais de 3000 drogarias em 7 países europeus. Atualmente é exigido que os gerentes de loja da Rossman façam a previsão de suas vendas diárias com antecipação de até 6 seis semanas. As vendas são influenciadas por muitos fatores, incluindo promoções, competições, feriados escolares e estaduais, fatores sazonais e localidade. Com milhares de gerentes prevendo suas vendas baseados em suas circunstâncias particulares, a precisão dos resultados pode ser bem variada.
#
# Em sua primeira competição Kaggle, a rede Rossman nos desafia a prever 6 semanas de vendas diárias para suas 1115 lojas espalhadas pela Alemanha. A previsão confiável das vendas capacita os gerentes de loja a criar plantões eficazes aumentando a produtividade e motivação dos empregados. Ao ajudar a rede Rossman a criar um modelo preditivo robusto, você irá ajudar os gerentes de loja a focarem no que é mais importante pra eles: seus consumidores e times!

# %% [markdown]
# ## Configuração Inicial
# Primeiramente vamos definir alguns parâmetros úteis de configuração do notebook, através do operador `%`. Este é um comando *mágico* do kernel IPython, o kernel utilizado pelo Jupyter Notebook na linguagem python, estes tipos de comando são específicos de cada kernel e normalmente definem parâmetros de configuração do kernel, Como o símbolo `%` não é um operador unário válida em python (ou seja, não é válido se utilizado sozinho na frente de uma expressão) ele é o utilizado por padrão para estes comando.

# %%
% matplotlib inline
% reload_ext autoreload
% autoreload 2

# %% [markdown]
# - `% matplotib inline` Mostrar os gráficos gerados pela biblioteca `matplotlib`, a maioria no geral, visto que a biblioteca `matplotlib` é usada como backend em outras bibliotecas como a `seaborn`
# - `% reload_ext autoreload` Carrega a extensão `autoreload`, essencialmente essa extensão recarrega as bibliotecas e scripts que foram importados na sua sessão atual após alguma alteração ter sido feita neles. Digamos que você está experimentando os métodos de determinada classe e no decorrer da escrita do código você precisa modificar o código dessa classe, seja refatorar o código, modificar as funções ou corrigir erros. Após tais modificações serem efetuadas, basta retornar para o notebook e prosseguir na experimentação, sem necessidade de reiniciar o kernel e importar novamente os módulos. 
# - `% autoreload 2` ativa a extensão carregada na linha anterior

# %%
PATH = '../data/competitions/rossmann/'

# %%
from fastai.structured import add_datepart
#from fastai.column_data import 

# %% [markdown]
# ## Importando o conjunto de dados

# %% [markdown]
# No desafio é fornecido os seguintes conjuntos de dados das 1.115 lojas da rede Rossman. Nosso objetivo é prever a coluna `Sales` (*Vendas*) no conjunto teste. Note que algumas lojas da rede foram temporariamente fechadas para reformas.
#
# Arquivos
#
#  - `train.csv` - dados históricos incluindo vendas.
#  - `test.csv` - dados históricos excluindo vendas ( o que queremos prever )
#  - `sample_submission.csv` - um arquivos de exemplo como submeter uma solução para o problema na forma correta
#  - `store.csv` - informação suplementar sobre as lojas
#
# Campos de dados
#
# A maioria das colunas é auto-explicativa. As seguintes são menos intuitivas.
#
#  - Id - um identificador que representa a tupla (Store, Date) tupla dentro do conjunto teste
#  - Store - Identificador único de cada loja 
#  - Sales - a receita do dia (Isto é o que estamos tentando prever)
#  - Customers - O número de consumidores em dado dia
#  - Open - Um indicador do estado da loja: 0 = closed (*fechada*), 1 = open (*aberta*)
#  - StateHoliday - indica um feriado estadual. Normalmente todas as lojas, com poucas exceções,  estão fechadas em feriados estaduais. Note que todas as escolas são fechadas nos feriados públicos. 
#    - a = public holiday (*feriado público*) 
#    - b = Easter holiday (*feriado de páscoa*)
#    - c = Christmas      (*Natal*)
#    - 0 = None           (*Não é feriado*) 
#  - SchoolHoliday - indica se a tupla (Store, Date) foi afetada pelo fechamento das escolas públicas.
#  - StoreType - diferencia entre os 4 modelos diferentes de lojas: a, b, c, d
#  - Assortment - descreve a variedade da loja: 
#    - a = basic ( *básica* )
#    - b = extra ( *extra* )
#    - c = extended ( *extendida* )
#  - CompetitionDistance - distância em metros da loja rival mais próxima
#  - CompetitionOpenSince[Month/Year] - fornece o ano e mês aproximado que a loja rival foi aberta
#     Promo - indica se a loja está com promoção ativa naquele dia
#     Promo2 - Promo2 indica se a presença de alguma promoção contínua e consecutiva para algumas lojas: 
#    - 0 = loja não está participando
#    - 1 = loja está participando
#  - Promo2Since[Year/Week] - descreve o ano e a semana na qual a loja começou a participar da Promo2
#  - PromoInterval - descreve os intervalos consecutivos de validade da Promo2, nomeando os meses nas quais a promoção é renovada. Por exemplo: "Feb,May,Aug,Nov" significa que as rodadas começaram em Fevereiro, Maio, Agosto e Novembro de cada ano na loja participante.

# %% [markdown]
# ## Conjuntos de Dados Adicionais
# Ao longo da competição, outros usuários forneceram conjuntos de dados adicionais que úteis

# %% [markdown]
# * `store_states`: mapeia a loja à unidade federativa alemã
# * `state_names`: Lista de Nomes dos Estados Alemães
# * `googletrend`: tendência de certas palavras-chave de busca ao longo do tempo no Google. Nos fóruns do Kaggle alguns usuários descobriram que se correlacionam bem com os dados do problema.
# * `weather`: Dados meteorológicos

# %%
