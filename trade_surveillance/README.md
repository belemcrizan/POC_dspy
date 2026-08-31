# Trade Surveillance Analítico para AWS/Jupyter

Pipeline batch, explicável e auditável para priorização de operações que exigem revisão de Surveillance.

> O score organiza uma fila investigativa. Ele não constitui prova de fraude, abuso de mercado ou cross trade.

## Conteúdo

- `Trade_Surveillance_Analitico_AWS.ipynb`: notebook principal, documentado célula a célula.
- `trade_surveillance_analitico.py`: versão equivalente em Python com células `# %%`.
- `requirements.txt`: dependências do módulo.

## Capacidades

- validação de schema, normalização e quarentena;
- benchmark histórico por peer group sem look-ahead;
- outliers robustos de taxa por mediana e IQR;
- outliers de volume em escala logarítmica;
- concentração por trader, ticker e estratégia;
- HHI, rajadas temporais, horário e possíveis duplicidades;
- candidato a cross trade com linguagem investigativa apropriada;
- score de 0 a 100 com motivos e contribuições rastreáveis;
- IC95%, análise de sensibilidade e backtest opcional com labels;
- dashboards Plotly e exportação Excel, Parquet/CSV e JSON;
- upload opcional para S3 usando a IAM Role do ambiente.

## Execução no Jupyter/AWS

Instale as dependências:

```python
%pip install -r trade_surveillance/requirements.txt
```

Reinicie o kernel, se solicitado. Carregue um DataFrame chamado `df` e execute o notebook:

```python
df = carregar_dados("s3://bucket/prefixo/operacoes.parquet")
```

Também é possível carregar a base antes de abrir o notebook:

```python
import pandas as pd

df = pd.read_parquet("s3://bucket/prefixo/operacoes.parquet")
```

## Colunas mínimas

| Coluna | Uso |
|---|---|
| `data_hora_negociacao` | ordenação temporal e regras de horário |
| `codigo_ticker` | peer group inicial |
| `valor_taxa_juro_operacao` | benchmark e desvio da taxa |
| `valor_financeiro` | volume, concentração e materialidade |

As demais colunas são opcionais e ativam sinais adicionais.

## Configuração crítica

Os limites padrão de taxa estão em pontos percentuais. Se `0.105` representar 10,5% na fonte, ajuste os thresholds.

Comparar apenas por ticker é um ponto de partida. Em uma base produtiva, use vencimento, indexador, moeda, rating e demais dimensões econômicas disponíveis:

```python
CONFIG = SurveillanceConfig(
    peer_cols=(
        "codigo_ticker",
        "data_vencimento",
        "codigo_indexador",
        "codigo_moeda",
    )
)
```

## Saídas

A execução cria `trade_surveillance_output/` com:

- workbook Excel com resumo, alertas, qualidade, quarentena e análises;
- alertas em CSV compactado;
- resultado completo em Parquet ou CSV.GZ;
- manifesto JSON com `run_id`, versão do score e hash da configuração.

Arquivos de saída e dados reais não devem ser versionados no Git.

## Validação

O pipeline foi exercitado com dados sintéticos contendo outliers de taxa e volume, duplicidade e registro inválido. Antes do uso produtivo, faça backtesting temporal com casos encerrados e obtenha aprovação formal dos limites pelo dono do controle.
