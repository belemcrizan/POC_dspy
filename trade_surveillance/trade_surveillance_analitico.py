# %% [markdown]
# # Trade Surveillance — análise robusta, explicável e auditável
#
# Este notebook transforma operações de renda fixa em uma fila priorizada de revisão.
# Ele foi desenhado para execução em Jupyter no AWS SageMaker Studio, EMR Notebook ou
# ambiente equivalente.
#
# **O que ele entrega**
#
# - validação do schema e quarentena de registros inválidos;
# - benchmark histórico por grupo comparável, usando somente operações anteriores;
# - outliers robustos de taxa e volume, menos sensíveis a caudas e valores extremos;
# - concentração por trader, estratégia e trader dentro de cada ticker;
# - candidatos a operação interna/cross trade, rajadas e operações fora do horário;
# - score de 0 a 100 com contribuição e justificativa de cada regra;
# - indicadores executivos, intervalos de confiança e análise de sensibilidade;
# - dashboards Plotly e exportação para Excel, Parquet/CSV e JSON de auditoria.
#
# > **Limite importante:** o score prioriza investigação. Ele não prova fraude, abuso de
# > mercado ou cross trade. Toda decisão material deve passar por evidência adicional e
# > revisão humana.

# %% [markdown]
# ## 1. Dependências
#
# Na maioria dos kernels AWS, `pandas` e `numpy` já estão disponíveis. Se necessário,
# execute uma única vez em uma célula separada:
#
# ```python
# %pip install -q pandas numpy plotly openpyxl pyarrow s3fs boto3
# ```

# %%
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_DISPONIVEL = True
except ImportError:
    PLOTLY_DISPONIVEL = False

try:
    from IPython.display import display
except ImportError:
    display = print

pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# %% [markdown]
# ## 2. Configuração central
#
# Os limites abaixo são **hipóteses de controle**, não verdades universais. Antes de uso
# produtivo, calibre-os por produto, mesa, vencimento, liquidez e histórico de casos.
#
# A unidade padrão da taxa é ponto percentual. Assim, `0.25` significa desvio de
# 0,25 p.p. Se a base armazenar taxa decimal (`0.105` para 10,5%), ajuste os limites.

# %%
@dataclass(frozen=True)
class SurveillanceConfig:
    # Colunas principais
    col_data: str = "data_hora_negociacao"
    col_ticker: str = "codigo_ticker"
    col_taxa: str = "valor_taxa_juro_operacao"
    col_financeiro: str = "valor_financeiro"
    col_quantidade: str = "valor_quantidade"
    col_trader: str = "numero_funcional_responsavel_operacao_parte"
    col_estrategia: str = "codigo_estrategia_parte"
    col_estrategia_contraparte: str = "codigo_estrategia_contraparte"
    col_lado: str = "descricao_lado_operacao_parte"
    col_operacao_interna: str = "indicador_operacao_interna"
    col_id_operacao: str = "id_operacao"
    col_label_confirmado: str = "flag_caso_confirmado"

    # Grupo econômico comparável. Adicione vencimento/indexador/moeda, se existirem.
    peer_cols: tuple[str, ...] = ("codigo_ticker",)

    # Interpretação dos dados
    input_timezone: str = "America/Sao_Paulo"
    decimal_comma: bool = True
    random_seed: int = 42

    # Benchmark histórico sem look-ahead
    rolling_trades: int = 120
    min_history: int = 20
    rate_iqr_floor: float = 0.02
    volume_log_iqr_floor: float = 0.10

    # Taxa — unidade padrão: ponto percentual
    rate_abs_warn: float = 0.25
    rate_abs_full_score: float = 0.75
    rate_min_material: float = 0.10
    rate_robust_z_warn: float = 3.5
    rate_robust_z_full_score: float = 7.0

    # Volume — z robusto aplicado sobre log(1 + financeiro absoluto)
    volume_robust_z_warn: float = 4.0
    volume_robust_z_full_score: float = 8.0

    # Concentração em janela batch
    trader_share_warn: float = 0.20
    trader_ticker_share_warn: float = 0.50
    strategy_share_warn: float = 0.30
    min_trades_for_ticker_concentration: int = 5

    # Padrões temporais
    burst_window: str = "30min"
    burst_min_trades: int = 5
    business_hour_start: int = 9
    business_hour_end: int = 18
    flag_weekends_as_off_hours: bool = True
    enable_off_hours_rule: bool = True

    # Severidade e operação
    alert_score_min: float = 40.0
    max_plot_rows: int = 50_000
    output_dir: str = "trade_surveillance_output"
    export_parquet: bool = True
    s3_output_uri: str | None = None


CONFIG = SurveillanceConfig()
CONFIG


# %% [markdown]
# ### Como tornar o peer group mais rigoroso
#
# Comparar somente por ticker pode misturar operações economicamente diferentes. Se a
# base tiver as colunas correspondentes, prefira algo como:
#
# ```python
# CONFIG = SurveillanceConfig(
#     peer_cols=("codigo_ticker", "data_vencimento", "codigo_indexador", "codigo_moeda")
# )
# ```
#
# Para mercados com spread sistemático entre compra e venda, considere adicionar também
# `descricao_lado_operacao_parte` ao peer group.

# %% [markdown]
# ## 3. Entrada opcional de dados
#
# Se `df` já existe na memória, não é preciso carregar nada. O helper abaixo aceita CSV,
# Parquet e Excel, inclusive URI `s3://` quando `s3fs`/`pyarrow` estão instalados.

# %%
def carregar_dados(uri: str, **kwargs: Any) -> pd.DataFrame:
    """Carrega uma fonte tabular local ou S3 sem embutir credenciais no notebook."""
    caminho = uri.lower().split("?", 1)[0]
    if caminho.endswith((".parquet", ".pq")):
        return pd.read_parquet(uri, **kwargs)
    if caminho.endswith(".csv") or caminho.endswith(".csv.gz"):
        return pd.read_csv(uri, **kwargs)
    if caminho.endswith((".xlsx", ".xls")):
        return pd.read_excel(uri, **kwargs)
    raise ValueError("Formato não reconhecido. Use CSV, Parquet ou Excel.")


# Exemplo:
# DATA_URI = "s3://meu-bucket/surveillance/operacoes.parquet"
# df = carregar_dados(DATA_URI)


# %% [markdown]
# ## 4. Funções de qualidade e preparação
#
# A rotina mantém três objetos distintos:
#
# - `dados_brutos`: cópia imutável da entrada;
# - `base`: registros elegíveis para a análise;
# - `quarentena`: registros inválidos, preservados para correção e auditoria.

# %%
def _coerce_numeric(series: pd.Series, decimal_comma: bool) -> pd.Series:
    """Converte números e tolera a notação brasileira quando configurada."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    texto = series.astype("string").str.strip()
    texto = texto.str.replace(r"[^0-9,\.\-+eE]", "", regex=True)

    if decimal_comma:
        tem_virgula = texto.str.contains(",", na=False)
        texto_br = (
            texto.loc[tem_virgula]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        texto = texto.copy()
        texto.loc[tem_virgula] = texto_br

    return pd.to_numeric(texto, errors="coerce")


def _parse_timestamp(series: pd.Series, input_timezone: str) -> pd.Series:
    """Interpreta datas sem timezone como horário local e retorna UTC."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce")

    try:
        tz_atual = parsed.dt.tz
    except AttributeError:
        # Mistura de timestamps aware/naive: fallback conservador para UTC.
        return pd.to_datetime(series, errors="coerce", utc=True)

    if tz_atual is None:
        return (
            parsed.dt.tz_localize(
                input_timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
            .dt.tz_convert("UTC")
        )
    return parsed.dt.tz_convert("UTC")


def _normalizar_booleano(series: pd.Series) -> pd.Series:
    mapa = {
        "true": True,
        "t": True,
        "1": True,
        "sim": True,
        "s": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "nao": False,
        "não": False,
        "n": False,
        "no": False,
    }
    normalizada = series.astype("string").str.strip().str.lower().map(mapa)
    return normalizada.astype("boolean")


def _fingerprint_dataframe(frame: pd.DataFrame, cols: list[str]) -> str:
    """Fingerprint reprodutível de uma amostra determinística, sem expor os dados."""
    if frame.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    cols_validas = [c for c in cols if c in frame.columns]
    amostra = pd.concat([frame[cols_validas].head(5_000), frame[cols_validas].tail(5_000)])
    hashes = pd.util.hash_pandas_object(amostra.astype("string"), index=True).values
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def validar_e_preparar(
    df_input: pd.DataFrame,
    cfg: SurveillanceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Valida schema, converte tipos e separa registros válidos da quarentena."""
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("A entrada precisa ser um pandas.DataFrame.")
    if df_input.empty:
        raise ValueError("O DataFrame está vazio.")

    obrigatorias = [cfg.col_data, cfg.col_ticker, cfg.col_taxa, cfg.col_financeiro]
    ausentes = [c for c in obrigatorias if c not in df_input.columns]
    if ausentes:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(ausentes)
        )

    peer_ausentes = [c for c in cfg.peer_cols if c not in df_input.columns]
    if peer_ausentes:
        raise ValueError(
            "Colunas do peer group ausentes: " + ", ".join(peer_ausentes)
        )

    dados = df_input.copy(deep=True)
    dados["_ordem_original"] = np.arange(len(dados), dtype=np.int64)

    # Mantém os valores originais dos campos críticos para diagnóstico da quarentena.
    for col in [cfg.col_data, cfg.col_taxa, cfg.col_financeiro]:
        dados[f"{col}__raw"] = dados[col]

    opcionais = [
        cfg.col_quantidade,
        cfg.col_trader,
        cfg.col_estrategia,
        cfg.col_estrategia_contraparte,
        cfg.col_lado,
        cfg.col_operacao_interna,
    ]
    for col in opcionais:
        if col not in dados.columns:
            dados[col] = pd.NA

    if cfg.col_id_operacao not in dados.columns:
        dados[cfg.col_id_operacao] = "ROW-" + dados["_ordem_original"].astype(str)
    else:
        id_fallback = "ROW-" + dados["_ordem_original"].astype(str)
        dados[cfg.col_id_operacao] = (
            dados[cfg.col_id_operacao].astype("string").fillna(id_fallback)
        )

    orig_data_nao_nula = dados[cfg.col_data].notna()
    orig_taxa_nao_nula = dados[cfg.col_taxa].notna()
    orig_fin_nao_nulo = dados[cfg.col_financeiro].notna()

    dados[cfg.col_data] = _parse_timestamp(dados[cfg.col_data], cfg.input_timezone)
    dados[cfg.col_taxa] = _coerce_numeric(dados[cfg.col_taxa], cfg.decimal_comma)
    dados[cfg.col_financeiro] = _coerce_numeric(
        dados[cfg.col_financeiro], cfg.decimal_comma
    )
    dados[cfg.col_quantidade] = _coerce_numeric(
        dados[cfg.col_quantidade], cfg.decimal_comma
    )
    dados["valor_financeiro_abs"] = dados[cfg.col_financeiro].abs()

    ticker_texto = dados[cfg.col_ticker].astype("string").str.strip()
    dados[cfg.col_ticker] = ticker_texto.replace("", pd.NA)

    interno_norm = _normalizar_booleano(dados[cfg.col_operacao_interna])
    dados["flag_indicador_interno_desconhecido"] = (
        dados[cfg.col_operacao_interna].notna() & interno_norm.isna()
    )
    dados["flag_operacao_interna"] = interno_norm.fillna(False).astype(bool)
    # O indicador interno é um proxy. Não é evidência suficiente para afirmar cross trade.
    dados["flag_cross_trade_candidato"] = dados["flag_operacao_interna"]

    duplicidade_subset = [
        cfg.col_data,
        cfg.col_ticker,
        cfg.col_taxa,
        cfg.col_financeiro,
        cfg.col_trader,
        cfg.col_lado,
    ]
    dados["flag_possivel_duplicidade"] = dados.duplicated(
        subset=duplicidade_subset, keep=False
    )

    dados["flag_registro_valido"] = (
        dados[cfg.col_data].notna()
        & dados[cfg.col_ticker].notna()
        & dados[cfg.col_taxa].notna()
        & dados[cfg.col_financeiro].notna()
        & dados["valor_financeiro_abs"].gt(0)
    )

    motivos_quarentena = pd.Series("", index=dados.index, dtype="string")

    def add_motivo(mask: pd.Series, texto: str) -> None:
        nonlocal motivos_quarentena
        # Evita soma de arrays NumPy de strings com larguras diferentes
        # (UFuncTypeError em ambientes com NumPy 2.x). A concatenação fica
        # inteiramente no dtype StringDtype do pandas.
        selecionados = mask.fillna(False).astype(bool)
        atuais = motivos_quarentena.loc[selecionados].fillna("")
        novo_motivo = pd.Series(texto, index=atuais.index, dtype="string")
        motivos_quarentena.loc[selecionados] = (
            atuais.str.cat(novo_motivo, sep="|").str.lstrip("|")
        )

    add_motivo(dados[cfg.col_data].isna(), "DATA_INVALIDA")
    add_motivo(dados[cfg.col_ticker].isna(), "TICKER_AUSENTE")
    add_motivo(dados[cfg.col_taxa].isna(), "TAXA_INVALIDA")
    add_motivo(dados[cfg.col_financeiro].isna(), "FINANCEIRO_INVALIDO")
    add_motivo(dados["valor_financeiro_abs"].le(0), "FINANCEIRO_NAO_POSITIVO")
    dados["motivo_quarentena"] = motivos_quarentena

    quarentena = dados.loc[~dados["flag_registro_valido"]].copy()
    base = dados.loc[dados["flag_registro_valido"]].copy()
    base = base.sort_values([cfg.col_data, "_ordem_original"]).reset_index(drop=True)

    qualidade = pd.DataFrame(
        [
            ("linhas_entrada", len(dados), "Total recebido"),
            ("linhas_validas", len(base), "Elegíveis para o score"),
            ("linhas_quarentena", len(quarentena), "Excluídas do score, não apagadas"),
            (
                "falha_conversao_data",
                int((orig_data_nao_nula & dados[cfg.col_data].isna()).sum()),
                "Valor original não nulo que não pôde ser interpretado",
            ),
            (
                "falha_conversao_taxa",
                int((orig_taxa_nao_nula & dados[cfg.col_taxa].isna()).sum()),
                "Valor original não nulo que não pôde ser interpretado",
            ),
            (
                "falha_conversao_financeiro",
                int((orig_fin_nao_nulo & dados[cfg.col_financeiro].isna()).sum()),
                "Valor original não nulo que não pôde ser interpretado",
            ),
            (
                "possiveis_duplicidades",
                int(dados["flag_possivel_duplicidade"].sum()),
                "Candidatos; operações legítimas podem ter campos iguais",
            ),
            (
                "indicador_interno_desconhecido",
                int(dados["flag_indicador_interno_desconhecido"].sum()),
                "Valores booleanos não reconhecidos",
            ),
        ],
        columns=["metrica", "valor", "interpretacao"],
    )

    metadata = {
        "linhas_entrada": int(len(dados)),
        "linhas_validas": int(len(base)),
        "linhas_quarentena": int(len(quarentena)),
        "fingerprint_amostra": _fingerprint_dataframe(
            dados,
            [cfg.col_id_operacao, cfg.col_data, cfg.col_ticker, cfg.col_taxa, cfg.col_financeiro],
        ),
    }
    return base, quarentena, qualidade, metadata


# %% [markdown]
# ## 5. Benchmark histórico e outliers robustos
#
# O código original calculava a mediana em toda a amostra. Em uma simulação histórica,
# isso deixa uma operação do futuro influenciar o benchmark do passado (*look-ahead*).
# Aqui, cada linha enxerga somente as `rolling_trades` operações anteriores do mesmo peer
# group.
#
# Usamos mediana e IQR, em vez de média e desvio-padrão. Isso é mais estável em dados de
# negociação, que frequentemente têm caudas pesadas. A aproximação de escala robusta é:
#
# `sigma_robusto = IQR / 1.349`

# %%
def _rolling_peer_stat(
    frame: pd.DataFrame,
    value_col: str,
    peer_cols: list[str],
    window: int,
    min_history: int,
) -> pd.DataFrame:
    grupos = frame.groupby(peer_cols, dropna=False, sort=False)[value_col]

    historico_n = grupos.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).count()
    )
    mediana = grupos.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_history).median()
    )
    q1 = grupos.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_history).quantile(0.25)
    )
    q3 = grupos.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_history).quantile(0.75)
    )
    return pd.DataFrame(
        {"historico_n": historico_n, "mediana": mediana, "q1": q1, "q3": q3},
        index=frame.index,
    )


def adicionar_benchmarks_historicos(
    base: pd.DataFrame,
    cfg: SurveillanceConfig,
) -> pd.DataFrame:
    out = base.copy()
    peer_cols = list(cfg.peer_cols)

    taxa_stats = _rolling_peer_stat(
        out,
        cfg.col_taxa,
        peer_cols,
        cfg.rolling_trades,
        cfg.min_history,
    )
    out["n_historico_taxa"] = taxa_stats["historico_n"]
    out["taxa_mercado_historica"] = taxa_stats["mediana"]
    out["taxa_q1_historica"] = taxa_stats["q1"]
    out["taxa_q3_historica"] = taxa_stats["q3"]
    out["taxa_mediana_descritiva_amostra"] = out.groupby(
        peer_cols, dropna=False, sort=False
    )[cfg.col_taxa].transform("median")

    out["flag_benchmark_taxa_suficiente"] = (
        out["n_historico_taxa"].ge(cfg.min_history)
        & out["taxa_mercado_historica"].notna()
    )
    out["desvio_taxa"] = out[cfg.col_taxa] - out["taxa_mercado_historica"]
    out["desvio_taxa_abs"] = out["desvio_taxa"].abs()
    escala_taxa = (
        (out["taxa_q3_historica"] - out["taxa_q1_historica"]) / 1.349
    ).clip(lower=cfg.rate_iqr_floor)
    out["escala_robusta_taxa"] = escala_taxa
    out["zscore_robusto_taxa"] = out["desvio_taxa"] / escala_taxa
    out["flag_taxa_desvio_absoluto"] = (
        out["flag_benchmark_taxa_suficiente"]
        & out["desvio_taxa_abs"].ge(cfg.rate_abs_warn)
    )
    out["flag_taxa_outlier_robusto"] = (
        out["flag_benchmark_taxa_suficiente"]
        & out["desvio_taxa_abs"].ge(cfg.rate_min_material)
        & out["zscore_robusto_taxa"].abs().ge(cfg.rate_robust_z_warn)
    )

    out["_log_volume"] = np.log1p(out["valor_financeiro_abs"])
    volume_stats = _rolling_peer_stat(
        out,
        "_log_volume",
        peer_cols,
        cfg.rolling_trades,
        cfg.min_history,
    )
    out["n_historico_volume"] = volume_stats["historico_n"]
    out["log_volume_mediano_historico"] = volume_stats["mediana"]
    out["volume_mediano_historico"] = np.expm1(
        out["log_volume_mediano_historico"]
    )
    escala_volume = (
        (volume_stats["q3"] - volume_stats["q1"]) / 1.349
    ).clip(lower=cfg.volume_log_iqr_floor)
    out["escala_robusta_log_volume"] = escala_volume
    out["zscore_robusto_volume"] = (
        out["_log_volume"] - out["log_volume_mediano_historico"]
    ) / escala_volume
    out["flag_benchmark_volume_suficiente"] = (
        out["n_historico_volume"].ge(cfg.min_history)
        & out["log_volume_mediano_historico"].notna()
    )
    # Para surveillance de tamanho, apenas a cauda superior é material.
    out["flag_volume_outlier_robusto"] = (
        out["flag_benchmark_volume_suficiente"]
        & out["zscore_robusto_volume"].ge(cfg.volume_robust_z_warn)
    )
    return out


# %% [markdown]
# ## 6. Concentração e comportamento temporal
#
# A concentração é calculada com financeiro absoluto para que sinais de compra/venda não
# se anulem. As métricas de concentração são contextuais ao período inteiro analisado;
# por isso são adequadas a uma rotina batch/EOD. Em um motor online, calcule essas
# participações apenas com histórico anterior à operação.

# %%
def _contagem_rolling_por_entidade(
    frame: pd.DataFrame,
    entity_col: str,
    time_col: str,
    window: str,
) -> pd.Series:
    resultado = pd.Series(0.0, index=frame.index)
    elegiveis = frame[entity_col].notna() & frame[time_col].notna()
    for _, grupo in frame.loc[elegiveis].groupby(entity_col, sort=False):
        grupo = grupo.sort_values(time_col)
        marcador = pd.Series(
            1.0,
            index=pd.DatetimeIndex(grupo[time_col]),
        )
        contagem = marcador.rolling(window=window, closed="both").sum().to_numpy()
        resultado.loc[grupo.index] = contagem
    return resultado


def adicionar_concentracao_e_comportamento(
    base: pd.DataFrame,
    cfg: SurveillanceConfig,
) -> pd.DataFrame:
    out = base.copy()
    total_volume = out["valor_financeiro_abs"].sum()

    if out[cfg.col_trader].notna().any() and total_volume > 0:
        volume_trader = out.groupby(cfg.col_trader, dropna=False)[
            "valor_financeiro_abs"
        ].transform("sum")
        out["pct_volume_trader"] = volume_trader / total_volume
    else:
        out["pct_volume_trader"] = np.nan
    out["flag_concentracao_trader"] = (
        out[cfg.col_trader].notna()
        & out["pct_volume_trader"].ge(cfg.trader_share_warn)
    )

    volume_ticker = out.groupby(cfg.col_ticker, dropna=False)[
        "valor_financeiro_abs"
    ].transform("sum")
    n_ticker = out.groupby(cfg.col_ticker, dropna=False)[cfg.col_id_operacao].transform(
        "size"
    )
    out["n_operacoes_ticker"] = n_ticker
    if out[cfg.col_trader].notna().any():
        volume_trader_ticker = out.groupby(
            [cfg.col_ticker, cfg.col_trader], dropna=False
        )["valor_financeiro_abs"].transform("sum")
        out["pct_volume_trader_no_ticker"] = volume_trader_ticker / volume_ticker
    else:
        out["pct_volume_trader_no_ticker"] = np.nan
    out["flag_dominancia_trader_ticker"] = (
        out[cfg.col_trader].notna()
        & n_ticker.ge(cfg.min_trades_for_ticker_concentration)
        & out["pct_volume_trader_no_ticker"].ge(cfg.trader_ticker_share_warn)
    )

    if out[cfg.col_estrategia].notna().any() and total_volume > 0:
        volume_estrategia = out.groupby(cfg.col_estrategia, dropna=False)[
            "valor_financeiro_abs"
        ].transform("sum")
        out["pct_volume_estrategia"] = volume_estrategia / total_volume
    else:
        out["pct_volume_estrategia"] = np.nan
    out["flag_concentracao_estrategia"] = (
        out[cfg.col_estrategia].notna()
        & out["pct_volume_estrategia"].ge(cfg.strategy_share_warn)
    )

    out["operacoes_trader_na_janela"] = _contagem_rolling_por_entidade(
        out,
        cfg.col_trader,
        cfg.col_data,
        cfg.burst_window,
    )
    out["flag_rajada_operacoes"] = out["operacoes_trader_na_janela"].ge(
        cfg.burst_min_trades
    )

    data_local = out[cfg.col_data].dt.tz_convert(cfg.input_timezone)
    out["data_hora_local"] = data_local
    fora_hora = (
        data_local.dt.hour.lt(cfg.business_hour_start)
        | data_local.dt.hour.ge(cfg.business_hour_end)
    )
    if cfg.flag_weekends_as_off_hours:
        fora_hora = fora_hora | data_local.dt.dayofweek.ge(5)
    out["flag_fora_horario"] = (
        fora_hora if cfg.enable_off_hours_rule else False
    )
    return out


# %% [markdown]
# ## 7. Score explicável
#
# O score é aditivo por famílias independentes, com limites para reduzir dupla contagem:
#
# | Família | Pontos máximos | Interpretação |
# |---|---:|---|
# | Taxa fora do benchmark | 45 | desvio absoluto + z robusto |
# | Volume atípico | 15 | cauda superior do log-volume |
# | Operação interna | 10 | candidato a cross trade; requer reconciliação |
# | Concentração | 15 | trader global/local e estratégia |
# | Rajada temporal | 6 | várias operações na mesma janela |
# | Fora do horário | 4 | regra operacional configurável |
# | Possível duplicidade | 5 | candidato a falha operacional/dado |
# | **Total** | **100** | fila de priorização, não probabilidade |

# %%
def _pontos_progressivos(
    valor: pd.Series,
    inicio: float,
    pontuacao_completa: float,
    max_pontos: float,
    mask: pd.Series | None = None,
) -> pd.Series:
    valor_num = pd.to_numeric(valor, errors="coerce").fillna(0.0)
    amplitude = max(pontuacao_completa - inicio, np.finfo(float).eps)
    progresso = ((valor_num - inicio) / amplitude).clip(0.0, 1.0)
    pontos = np.where(
        valor_num.ge(inicio),
        max_pontos * (0.25 + 0.75 * progresso),
        0.0,
    )
    resultado = pd.Series(pontos, index=valor.index, dtype=float)
    if mask is not None:
        resultado = resultado.where(mask.fillna(False), 0.0)
    return resultado


def calcular_score_explicavel(
    base: pd.DataFrame,
    cfg: SurveillanceConfig,
) -> pd.DataFrame:
    out = base.copy()

    out["pontos_taxa_desvio"] = _pontos_progressivos(
        out["desvio_taxa_abs"],
        cfg.rate_abs_warn,
        cfg.rate_abs_full_score,
        25.0,
        out["flag_benchmark_taxa_suficiente"],
    )
    out["pontos_taxa_robusta"] = _pontos_progressivos(
        out["zscore_robusto_taxa"].abs(),
        cfg.rate_robust_z_warn,
        cfg.rate_robust_z_full_score,
        20.0,
        out["flag_taxa_outlier_robusto"],
    )
    out["pontos_volume"] = _pontos_progressivos(
        out["zscore_robusto_volume"],
        cfg.volume_robust_z_warn,
        cfg.volume_robust_z_full_score,
        15.0,
        out["flag_volume_outlier_robusto"],
    )
    out["pontos_operacao_interna"] = np.where(
        out["flag_cross_trade_candidato"], 10.0, 0.0
    )

    pontos_trader_global = _pontos_progressivos(
        out["pct_volume_trader"],
        cfg.trader_share_warn,
        min(1.0, cfg.trader_share_warn * 2.5),
        8.0,
        out["flag_concentracao_trader"],
    )
    pontos_trader_ticker = _pontos_progressivos(
        out["pct_volume_trader_no_ticker"],
        cfg.trader_ticker_share_warn,
        1.0,
        10.0,
        out["flag_dominancia_trader_ticker"],
    )
    # Usa o maior sinal de trader, não a soma, para evitar dupla contagem.
    out["pontos_concentracao_trader"] = np.maximum(
        pontos_trader_global, pontos_trader_ticker
    )
    out["pontos_concentracao_estrategia"] = _pontos_progressivos(
        out["pct_volume_estrategia"],
        cfg.strategy_share_warn,
        min(1.0, cfg.strategy_share_warn * 2.0),
        5.0,
        out["flag_concentracao_estrategia"],
    )
    out["pontos_rajada"] = np.where(out["flag_rajada_operacoes"], 6.0, 0.0)
    out["pontos_fora_horario"] = np.where(out["flag_fora_horario"], 4.0, 0.0)
    out["pontos_duplicidade"] = np.where(
        out["flag_possivel_duplicidade"], 5.0, 0.0
    )

    colunas_pontos = [
        "pontos_taxa_desvio",
        "pontos_taxa_robusta",
        "pontos_volume",
        "pontos_operacao_interna",
        "pontos_concentracao_trader",
        "pontos_concentracao_estrategia",
        "pontos_rajada",
        "pontos_fora_horario",
        "pontos_duplicidade",
    ]
    out["score"] = out[colunas_pontos].sum(axis=1).clip(0, 100).round(2)
    out["nivel_risco"] = pd.cut(
        out["score"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["BAIXO", "MEDIO", "ALTO", "CRITICO"],
        right=False,
    ).astype("string")
    out["flag_alerta"] = out["score"].ge(cfg.alert_score_min)

    out["motivos_alerta"] = ""
    out["explicacao_alerta"] = ""

    def adicionar_evidencia(mask: pd.Series, codigo: str, descricao: pd.Series | str) -> None:
        mask = mask.fillna(False)
        separador_codigo = np.where(out["motivos_alerta"].eq(""), "", "|")
        out.loc[mask, "motivos_alerta"] = (
            out.loc[mask, "motivos_alerta"]
            + pd.Series(separador_codigo, index=out.index).loc[mask]
            + codigo
        )

        if isinstance(descricao, str):
            desc_series = pd.Series(descricao, index=out.index)
        else:
            desc_series = descricao.astype("string")
        separador_desc = np.where(out["explicacao_alerta"].eq(""), "", " | ")
        out.loc[mask, "explicacao_alerta"] = (
            out.loc[mask, "explicacao_alerta"]
            + pd.Series(separador_desc, index=out.index).loc[mask]
            + desc_series.loc[mask]
        )

    desc_taxa = (
        "Taxa="
        + out[cfg.col_taxa].round(4).astype("string")
        + ", benchmark histórico="
        + out["taxa_mercado_historica"].round(4).astype("string")
        + ", desvio="
        + out["desvio_taxa"].round(4).astype("string")
        + " p.p."
    )
    adicionar_evidencia(out["flag_taxa_desvio_absoluto"], "TAXA_DESVIO_ABS", desc_taxa)
    adicionar_evidencia(
        out["flag_taxa_outlier_robusto"],
        "TAXA_OUTLIER_ROBUSTO",
        "Z robusto da taxa=" + out["zscore_robusto_taxa"].round(2).astype("string"),
    )
    adicionar_evidencia(
        out["flag_volume_outlier_robusto"],
        "VOLUME_OUTLIER_ROBUSTO",
        "Z robusto do log-volume="
        + out["zscore_robusto_volume"].round(2).astype("string"),
    )
    adicionar_evidencia(
        out["flag_cross_trade_candidato"],
        "OPERACAO_INTERNA_REVISAR_CROSS",
        "Indicador interno ativo; reconciliar as duas pontas e o beneficiário final",
    )
    adicionar_evidencia(
        out["flag_concentracao_trader"],
        "CONCENTRACAO_TRADER_GLOBAL",
        "Trader representa "
        + (100 * out["pct_volume_trader"]).round(2).astype("string")
        + "% do financeiro do período",
    )
    adicionar_evidencia(
        out["flag_dominancia_trader_ticker"],
        "DOMINANCIA_TRADER_TICKER",
        "Trader representa "
        + (100 * out["pct_volume_trader_no_ticker"]).round(2).astype("string")
        + "% do financeiro do ticker",
    )
    adicionar_evidencia(
        out["flag_concentracao_estrategia"],
        "CONCENTRACAO_ESTRATEGIA",
        "Estratégia representa "
        + (100 * out["pct_volume_estrategia"]).round(2).astype("string")
        + "% do financeiro do período",
    )
    adicionar_evidencia(
        out["flag_rajada_operacoes"],
        "RAJADA_OPERACOES",
        out["operacoes_trader_na_janela"].astype(int).astype("string")
        + f" operações do trader em {cfg.burst_window}",
    )
    adicionar_evidencia(
        out["flag_fora_horario"],
        "FORA_HORARIO",
        "Operação fora da janela operacional configurada",
    )
    adicionar_evidencia(
        out["flag_possivel_duplicidade"],
        "POSSIVEL_DUPLICIDADE",
        "Campos principais coincidem com outra linha; validar chave da operação",
    )

    flags_sinais = [
        "flag_taxa_desvio_absoluto",
        "flag_taxa_outlier_robusto",
        "flag_volume_outlier_robusto",
        "flag_cross_trade_candidato",
        "flag_concentracao_trader",
        "flag_dominancia_trader_ticker",
        "flag_concentracao_estrategia",
        "flag_rajada_operacoes",
        "flag_fora_horario",
        "flag_possivel_duplicidade",
    ]
    out["numero_sinais"] = out[flags_sinais].sum(axis=1).astype(int)
    out["explicacao_alerta"] = out["explicacao_alerta"].replace(
        "", "Nenhum sinal material pelas regras configuradas"
    )
    return out.drop(columns=["_log_volume"], errors="ignore")


# %% [markdown]
# ## 8. Sumários, concentração do portfólio e validação opcional

# %%
def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    margem = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centro - margem), min(1.0, centro + margem)


def _hhi(series: pd.Series) -> float:
    total = series.sum()
    if total <= 0:
        return np.nan
    shares = series / total
    return float((shares**2).sum())


def _sumario_por_coluna(
    frame: pd.DataFrame,
    coluna: str,
    cfg: SurveillanceConfig,
) -> pd.DataFrame:
    if coluna not in frame.columns or not frame[coluna].notna().any():
        return pd.DataFrame()
    resumo = (
        frame.loc[frame[coluna].notna()]
        .groupby(coluna, dropna=False)
        .agg(
            n_operacoes=(cfg.col_id_operacao, "size"),
            valor_financeiro=("valor_financeiro_abs", "sum"),
            n_alertas=("flag_alerta", "sum"),
            score_medio=("score", "mean"),
            score_maximo=("score", "max"),
            score_p95=("score", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    resumo["taxa_alerta"] = resumo["n_alertas"] / resumo["n_operacoes"]
    resumo["pct_financeiro"] = resumo["valor_financeiro"] / resumo[
        "valor_financeiro"
    ].sum()
    resumo["indice_risco"] = (
        0.50 * resumo["score_p95"]
        + 0.30 * resumo["score_maximo"]
        + 0.20 * resumo["score_medio"]
    )
    return resumo.sort_values(
        ["indice_risco", "valor_financeiro"], ascending=[False, False]
    )


def _avaliar_com_labels(
    frame: pd.DataFrame,
    cfg: SurveillanceConfig,
) -> pd.DataFrame:
    if cfg.col_label_confirmado not in frame.columns:
        return pd.DataFrame()
    y = _normalizar_booleano(frame[cfg.col_label_confirmado])
    validos = y.notna()
    if not validos.any():
        return pd.DataFrame()
    y = y.loc[validos].astype(bool)
    scores = frame.loc[validos, "score"]
    linhas = []
    for limiar in range(20, 91, 5):
        pred = scores.ge(limiar)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        tn = int((~pred & ~y).sum())
        precisao = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        especificidade = tn / (tn + fp) if tn + fp else np.nan
        f1 = (
            2 * precisao * recall / (precisao + recall)
            if precisao + recall and not (np.isnan(precisao) or np.isnan(recall))
            else np.nan
        )
        linhas.append(
            (limiar, tp, fp, fn, tn, precisao, recall, especificidade, f1)
        )
    return pd.DataFrame(
        linhas,
        columns=[
            "limiar",
            "tp",
            "fp",
            "fn",
            "tn",
            "precisao",
            "recall",
            "especificidade",
            "f1",
        ],
    )


def construir_sumarios(
    frame: pd.DataFrame,
    qualidade: pd.DataFrame,
    metadata: dict[str, Any],
    cfg: SurveillanceConfig,
) -> dict[str, pd.DataFrame]:
    n = len(frame)
    k = int(frame["flag_alerta"].sum())
    ci_inf, ci_sup = _wilson_interval(k, n)

    cobertura_taxa = float(frame["flag_benchmark_taxa_suficiente"].mean()) if n else np.nan
    resumo_executivo = pd.DataFrame(
        [
            ("Operações recebidas", metadata["linhas_entrada"], "linhas"),
            ("Operações analisadas", n, "linhas"),
            ("Registros em quarentena", metadata["linhas_quarentena"], "linhas"),
            ("Financeiro analisado", frame["valor_financeiro_abs"].sum(), "moeda da base"),
            ("Alertas", k, f"score >= {cfg.alert_score_min:g}"),
            ("Taxa de alertas", k / n if n else np.nan, "proporção"),
            ("IC95% taxa de alertas — inferior", ci_inf, "Wilson"),
            ("IC95% taxa de alertas — superior", ci_sup, "Wilson"),
            ("Críticos", int(frame["nivel_risco"].eq("CRITICO").sum()), "linhas"),
            ("Altos", int(frame["nivel_risco"].eq("ALTO").sum()), "linhas"),
            ("Médios", int(frame["nivel_risco"].eq("MEDIO").sum()), "linhas"),
            ("Cobertura benchmark taxa", cobertura_taxa, "proporção"),
            (
                "Operações internas candidatas a cross",
                int(frame["flag_cross_trade_candidato"].sum()),
                "exige investigação",
            ),
        ],
        columns=["indicador", "valor", "unidade_ou_nota"],
    )

    resumo_ticker = _sumario_por_coluna(frame, cfg.col_ticker, cfg)
    resumo_trader = _sumario_por_coluna(frame, cfg.col_trader, cfg)
    resumo_estrategia = _sumario_por_coluna(frame, cfg.col_estrategia, cfg)

    diario = (
        frame.assign(data=frame["data_hora_local"].dt.date)
        .groupby("data")
        .agg(
            n_operacoes=(cfg.col_id_operacao, "size"),
            valor_financeiro=("valor_financeiro_abs", "sum"),
            n_alertas=("flag_alerta", "sum"),
            score_medio=("score", "mean"),
            score_maximo=("score", "max"),
        )
        .reset_index()
    )
    diario["taxa_alerta"] = diario["n_alertas"] / diario["n_operacoes"]

    motivos = (
        frame.loc[frame["motivos_alerta"].ne(""), "motivos_alerta"]
        .str.get_dummies(sep="|")
        .sum()
        .sort_values(ascending=False)
        .rename_axis("motivo")
        .reset_index(name="n_operacoes")
    )

    sensibilidade = pd.DataFrame({"limiar_score": range(20, 91, 5)})
    sensibilidade["n_alertas"] = sensibilidade["limiar_score"].map(
        lambda t: int(frame["score"].ge(t).sum())
    )
    sensibilidade["taxa_alerta"] = sensibilidade["n_alertas"] / max(n, 1)

    concentracao = pd.DataFrame(
        [
            (
                "HHI trader",
                _hhi(
                    frame.groupby(cfg.col_trader, dropna=True)["valor_financeiro_abs"].sum()
                ),
                "Quanto maior, maior a concentração; interpretar com número de traders",
            ),
            (
                "HHI estratégia",
                _hhi(
                    frame.groupby(cfg.col_estrategia, dropna=True)["valor_financeiro_abs"].sum()
                ),
                "Quanto maior, maior a concentração; interpretar com mandato da mesa",
            ),
            (
                "HHI ticker",
                _hhi(
                    frame.groupby(cfg.col_ticker, dropna=True)["valor_financeiro_abs"].sum()
                ),
                "Concentração do financeiro por instrumento",
            ),
        ],
        columns=["metrica", "valor", "interpretacao"],
    )

    return {
        "resumo_executivo": resumo_executivo,
        "qualidade": qualidade,
        "resumo_ticker": resumo_ticker,
        "resumo_trader": resumo_trader,
        "resumo_estrategia": resumo_estrategia,
        "serie_diaria": diario,
        "motivos": motivos,
        "sensibilidade": sensibilidade,
        "concentracao": concentracao,
        "backtest_labels": _avaliar_com_labels(frame, cfg),
    }


# %% [markdown]
# ## 9. Orquestração reexecutável

# %%
def executar_surveillance(
    df_input: pd.DataFrame,
    cfg: SurveillanceConfig = CONFIG,
) -> dict[str, Any]:
    inicio = datetime.now(timezone.utc)
    base, quarentena, qualidade, metadata = validar_e_preparar(df_input, cfg)
    if base.empty:
        raise ValueError(
            "Nenhuma linha válida restou após a preparação. Consulte a quarentena."
        )

    base = adicionar_benchmarks_historicos(base, cfg)
    base = adicionar_concentracao_e_comportamento(base, cfg)
    resultado = calcular_score_explicavel(base, cfg)
    resultado = resultado.sort_values(
        ["score", cfg.col_data], ascending=[False, False]
    ).reset_index(drop=True)

    config_hash = hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    run_id = hashlib.sha256(
        f"{inicio.isoformat()}|{metadata['fingerprint_amostra']}|{config_hash}".encode()
    ).hexdigest()[:16]
    resultado["run_id"] = run_id
    resultado["score_model_version"] = "rules-v2-robust-historical"
    resultado["config_hash"] = config_hash

    alertas = resultado.loc[resultado["flag_alerta"]].copy()
    sumarios = construir_sumarios(resultado, qualidade, metadata, cfg)

    metadata.update(
        {
            "run_id": run_id,
            "inicio_utc": inicio.isoformat(),
            "fim_utc": datetime.now(timezone.utc).isoformat(),
            "peer_cols": list(cfg.peer_cols),
            "score_model": "rules-v2-robust-historical",
            "config_hash": config_hash,
            "alertas": int(len(alertas)),
        }
    )
    return {
        "df_resultado": resultado,
        "alertas": alertas,
        "quarentena": quarentena,
        "sumarios": sumarios,
        "metadata": metadata,
        "config": cfg,
    }


# %% [markdown]
# ## 10. Dashboards
#
# Para preservar memória, os gráficos usam amostragem apenas na camada visual. Todos os
# números, scores e tabelas continuam calculados sobre a base completa.

# %%
CORES_RISCO = {
    "BAIXO": "#2E8B57",
    "MEDIO": "#F2C14E",
    "ALTO": "#F78154",
    "CRITICO": "#C1292E",
}


def _amostra_visual(frame: pd.DataFrame, cfg: SurveillanceConfig) -> pd.DataFrame:
    if len(frame) <= cfg.max_plot_rows:
        return frame.copy()
    prioritarias = frame.loc[frame["flag_alerta"]]
    vagas = max(cfg.max_plot_rows - len(prioritarias), 0)
    baixas = frame.loc[~frame["flag_alerta"]]
    amostra_baixas = baixas.sample(
        n=min(vagas, len(baixas)), random_state=cfg.random_seed
    )
    return pd.concat([prioritarias, amostra_baixas]).drop_duplicates(
        subset=[cfg.col_id_operacao]
    )


def criar_dashboards(
    resultados: dict[str, Any],
    cfg: SurveillanceConfig = CONFIG,
) -> dict[str, Any]:
    if not PLOTLY_DISPONIVEL:
        warnings.warn(
            "Plotly não está instalado. Execute `%pip install plotly` e rode esta célula novamente."
        )
        return {}

    frame = resultados["df_resultado"]
    plot_df = _amostra_visual(frame, cfg)
    figs: dict[str, Any] = {}

    contagem = (
        frame["nivel_risco"]
        .value_counts()
        .reindex(["BAIXO", "MEDIO", "ALTO", "CRITICO"], fill_value=0)
        .rename_axis("nivel_risco")
        .reset_index(name="operacoes")
    )
    figs["distribuicao_risco"] = px.bar(
        contagem,
        x="nivel_risco",
        y="operacoes",
        color="nivel_risco",
        color_discrete_map=CORES_RISCO,
        category_orders={"nivel_risco": ["BAIXO", "MEDIO", "ALTO", "CRITICO"]},
        title="Operações por nível de risco",
        text_auto=True,
    )

    scatter_df = plot_df.loc[plot_df["flag_benchmark_taxa_suficiente"]].copy()
    if not scatter_df.empty:
        figs["taxa_vs_benchmark"] = px.scatter(
            scatter_df,
            x="taxa_mercado_historica",
            y=cfg.col_taxa,
            size="valor_financeiro_abs",
            size_max=35,
            color="nivel_risco",
            color_discrete_map=CORES_RISCO,
            hover_data=[
                cfg.col_id_operacao,
                cfg.col_ticker,
                cfg.col_trader,
                "desvio_taxa",
                "zscore_robusto_taxa",
                "score",
                "motivos_alerta",
            ],
            title="Taxa negociada versus benchmark histórico anterior",
            labels={
                "taxa_mercado_historica": "Benchmark histórico",
                cfg.col_taxa: "Taxa negociada",
            },
        )
        minimo = min(
            scatter_df["taxa_mercado_historica"].min(),
            scatter_df[cfg.col_taxa].min(),
        )
        maximo = max(
            scatter_df["taxa_mercado_historica"].max(),
            scatter_df[cfg.col_taxa].max(),
        )
        figs["taxa_vs_benchmark"].add_shape(
            type="line",
            x0=minimo,
            y0=minimo,
            x1=maximo,
            y1=maximo,
            line=dict(color="gray", dash="dash"),
        )

    diario = resultados["sumarios"]["serie_diaria"]
    fig_diario = make_subplots(specs=[[{"secondary_y": True}]])
    fig_diario.add_trace(
        go.Bar(
            x=diario["data"],
            y=diario["valor_financeiro"],
            name="Financeiro",
            marker_color="#4C78A8",
        ),
        secondary_y=False,
    )
    fig_diario.add_trace(
        go.Scatter(
            x=diario["data"],
            y=100 * diario["taxa_alerta"],
            name="Taxa de alertas (%)",
            mode="lines+markers",
            line=dict(color="#C1292E", width=3),
        ),
        secondary_y=True,
    )
    fig_diario.update_layout(title="Financeiro e taxa de alertas por dia")
    fig_diario.update_yaxes(title_text="Financeiro", secondary_y=False)
    fig_diario.update_yaxes(title_text="Taxa de alertas (%)", secondary_y=True)
    figs["evolucao_diaria"] = fig_diario

    tickers = resultados["sumarios"]["resumo_ticker"].head(20).sort_values(
        "indice_risco"
    )
    figs["tickers_risco"] = px.bar(
        tickers,
        x="indice_risco",
        y=cfg.col_ticker,
        orientation="h",
        color="taxa_alerta",
        hover_data=["n_operacoes", "n_alertas", "score_maximo", "valor_financeiro"],
        title="Top 20 tickers por índice de risco composto",
        color_continuous_scale="YlOrRd",
    )

    motivos = resultados["sumarios"]["motivos"].head(20).sort_values("n_operacoes")
    if not motivos.empty:
        figs["motivos"] = px.bar(
            motivos,
            x="n_operacoes",
            y="motivo",
            orientation="h",
            title="Frequência dos sinais de surveillance",
        )

    return figs


def mostrar_dashboards(figuras: dict[str, Any]) -> None:
    for figura in figuras.values():
        figura.show()


# %% [markdown]
# ## 11. Exportação auditável
#
# O Excel contém abas executivas e investigativas. O resultado completo é exportado em
# Parquet quando o engine estiver disponível; caso contrário, usa CSV compactado.
# Opcionalmente, os arquivos podem ser enviados a um prefixo S3 usando a role do kernel.

# %%
def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            out[col] = out[col].astype("string")
    return out


def _upload_s3(arquivos: list[Path], s3_uri: str) -> list[str]:
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("Instale boto3 para habilitar o upload S3.") from exc

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("s3_output_uri deve seguir o formato s3://bucket/prefixo/")
    bucket = parsed.netloc
    prefixo = parsed.path.strip("/")
    client = boto3.client("s3")
    destinos = []
    for arquivo in arquivos:
        chave = "/".join(filter(None, [prefixo, arquivo.name]))
        client.upload_file(str(arquivo), bucket, chave)
        destinos.append(f"s3://{bucket}/{chave}")
    return destinos


def exportar_resultados(
    resultados: dict[str, Any],
    cfg: SurveillanceConfig = CONFIG,
) -> dict[str, Any]:
    pasta = Path(cfg.output_dir)
    pasta.mkdir(parents=True, exist_ok=True)

    frame = resultados["df_resultado"]
    alertas = resultados["alertas"]
    quarentena = resultados["quarentena"]
    sumarios = resultados["sumarios"]

    caminho_excel = pasta / "trade_surveillance_alertas.xlsx"
    limite_excel = 1_000_000
    if len(alertas) > limite_excel:
        warnings.warn(
            f"Excel limitado às primeiras {limite_excel:,} linhas de alertas. "
            "O resultado completo permanece no arquivo analítico."
        )

    with pd.ExcelWriter(caminho_excel, engine="openpyxl") as writer:
        _excel_safe(sumarios["resumo_executivo"]).to_excel(
            writer, sheet_name="Resumo", index=False
        )
        _excel_safe(alertas.head(limite_excel)).to_excel(
            writer, sheet_name="Alertas", index=False
        )
        _excel_safe(frame.head(5_000)).to_excel(
            writer, sheet_name="Ranking_Top5000", index=False
        )
        _excel_safe(sumarios["qualidade"]).to_excel(
            writer, sheet_name="Qualidade", index=False
        )
        _excel_safe(quarentena.head(limite_excel)).to_excel(
            writer, sheet_name="Quarentena", index=False
        )
        config_tabela = pd.DataFrame(
            [
                (chave, json.dumps(valor, ensure_ascii=False, default=str))
                for chave, valor in asdict(cfg).items()
            ],
            columns=["parametro", "valor"],
        )
        config_tabela.to_excel(writer, sheet_name="Configuracao", index=False)
        for chave, aba in [
            ("resumo_ticker", "Tickers"),
            ("resumo_trader", "Traders"),
            ("resumo_estrategia", "Estrategias"),
            ("serie_diaria", "Serie_Diaria"),
            ("motivos", "Motivos"),
            ("sensibilidade", "Sensibilidade"),
            ("concentracao", "Concentracao"),
            ("backtest_labels", "Backtest_Labels"),
        ]:
            tabela = sumarios[chave]
            if not tabela.empty:
                _excel_safe(tabela).to_excel(writer, sheet_name=aba, index=False)

    arquivos = [caminho_excel]
    caminho_alertas_csv = pasta / "trade_surveillance_alertas.csv.gz"
    _excel_safe(alertas).to_csv(caminho_alertas_csv, index=False, compression="gzip")
    arquivos.append(caminho_alertas_csv)

    if cfg.export_parquet:
        caminho_analitico = pasta / "trade_surveillance_resultado.parquet"
        try:
            frame.to_parquet(caminho_analitico, index=False)
        except (ImportError, ValueError):
            caminho_analitico = pasta / "trade_surveillance_resultado.csv.gz"
            _excel_safe(frame).to_csv(
                caminho_analitico, index=False, compression="gzip"
            )
            warnings.warn(
                "Engine Parquet indisponível; resultado completo salvo como CSV.GZ."
            )
        arquivos.append(caminho_analitico)

    manifesto = {
        "metadata": resultados["metadata"],
        "config": asdict(cfg),
        "arquivos": [str(p) for p in arquivos],
        "aviso": "Score de priorização investigativa; não constitui prova de irregularidade.",
    }
    caminho_manifesto = pasta / "manifesto_execucao.json"
    caminho_manifesto.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    arquivos.append(caminho_manifesto)

    s3_destinos: list[str] = []
    if cfg.s3_output_uri:
        s3_destinos = _upload_s3(arquivos, cfg.s3_output_uri)

    return {
        "arquivos_locais": [str(p.resolve()) for p in arquivos],
        "destinos_s3": s3_destinos,
    }


# %% [markdown]
# ## 12. Execução
#
# Execute esta célula depois de definir `df`. Ela exibe o resumo, os 50 principais
# alertas, os gráficos e salva os artefatos.

# %%
if "df" not in globals():
    print(
        "Defina o DataFrame `df` ou carregue uma fonte com `carregar_dados(...)`; "
        "depois execute esta célula novamente."
    )
else:
    resultados = executar_surveillance(df, CONFIG)
    df_resultado = resultados["df_resultado"]
    alertas = resultados["alertas"]
    quarentena = resultados["quarentena"]

    print("RESUMO EXECUTIVO")
    display(resultados["sumarios"]["resumo_executivo"])

    colunas_top = [
        CONFIG.col_id_operacao,
        CONFIG.col_ticker,
        "data_hora_local",
        CONFIG.col_lado,
        CONFIG.col_taxa,
        "taxa_mercado_historica",
        "desvio_taxa",
        "zscore_robusto_taxa",
        "valor_financeiro_abs",
        "zscore_robusto_volume",
        "score",
        "nivel_risco",
        CONFIG.col_trader,
        CONFIG.col_estrategia,
        "motivos_alerta",
        "explicacao_alerta",
    ]
    print("TOP 50 ALERTAS")
    display(alertas[colunas_top].head(50))

    figuras = criar_dashboards(resultados, CONFIG)
    mostrar_dashboards(figuras)

    exportacao = exportar_resultados(resultados, CONFIG)
    print("ARQUIVOS GERADOS")
    for caminho in exportacao["arquivos_locais"]:
        print("-", caminho)
    for destino in exportacao["destinos_s3"]:
        print("-", destino)


# %% [markdown]
# ## 13. Como interpretar o resultado
#
# 1. **Comece pelo motivo, não pelo score.** O score ordena a fila; a explicação informa
#    qual evidência deve ser verificada.
# 2. **Cheque a cobertura do benchmark.** Linhas sem histórico mínimo não recebem pontos
#    de outlier de taxa/volume. Isso é uma proteção contra conclusões frágeis.
# 3. **Cross trade é apenas candidato.** Para confirmar, reconcilie compra e venda,
#    contraparte, beneficiário final, horário, quantidade, preço/taxa e intenção econômica.
# 4. **Concentração não é irregularidade por si.** Compare com mandato, alçadas, tamanho da
#    mesa, liquidez e concentração natural da carteira.
# 5. **Calibre com casos encerrados.** Se `flag_caso_confirmado` existir, a aba
#    `Backtest_Labels` mostra precisão, recall, especificidade e F1 por limiar.
# 6. **Monitore drift.** Mudanças persistentes em cobertura, taxa de alertas ou motivos
#    podem indicar alteração do mercado, do mix de produtos ou da qualidade da fonte.

# %% [markdown]
# ## 14. Evoluções recomendadas para produção
#
# - benchmark externo de curva/preço com timestamp e fonte de mercado;
# - vencimento, duration, indexador, rating, moeda e liquidez no peer group;
# - pareamento real das duas pernas para cross/wash trade;
# - grafo trader–cliente–contraparte–ativo–estratégia;
# - limites por produto/mesa aprovados pelo dono do controle;
# - backtesting temporal, controle de falsos positivos e revisão de thresholds;
# - versionamento da regra, trilha de decisão humana e SLA do caso;
# - execução distribuída em Spark/Glue para dezenas de milhões de linhas;
# - monitoramento de drift, cobertura e estabilidade do score em produção.
