"""Triagem explicável PF/PJ — núcleo determinístico, leitura da origem e revisão humana.

Uso rápido (sem AWS):
    python agente_alertas_robusto.py --demo

No SageMaker/JupyterLab:
    from agente_alertas_robusto import Config, AthenaRepository, AlertPipeline, run_all
    config = Config.from_json("config.example.json")
    relatorio = run_all(config, AthenaRepository.from_config(config))

Organização didática deste arquivo:
    1. Contratos/configuração; 2. Athena/Glue; 3. Qualidade; 4. Regras;
    5. Grafo/revisão/auditoria; 6. Avaliação offline; 7. CLI/demonstração.

O programa não encerra alertas. Quantidade de evidências não é probabilidade,
nem prova de ilícito. D0, D1 e delta são sinais estatisticamente relacionados.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

VERSION = "2.0.0"
LOGGER = logging.getLogger("alertas")
Z0 = "z_score_20dias_d0"
Z1 = "z_score_20dias_d1"
PRICE = "variacao_preco_d0_d1"
CUSTODY = "variacao_custodia_d0_d1"
NUMERIC_FIELDS = (Z0, Z1, PRICE, CUSTODY)
POST_EVENT_FIELDS = frozenset((Z1, PRICE, CUSTODY))
VALID_BASES = frozenset(("PF", "PJ"))


# -----------------------------------------------------------------------------
# 1. Contratos e configuração. Nenhum cliente AWS é criado durante o import.
# -----------------------------------------------------------------------------
class SchemaError(ValueError):
    """O contrato não permite interpretar este lote sem ambiguidade."""


class SourceError(RuntimeError):
    """Falha sanitizada de consulta; não inclui SQL, valores ou mensagem da AWS."""

    def __init__(self, code: str, query_id: str | None = None, **metadata: Any):
        super().__init__(code)
        self.code = code
        self.query_id = query_id
        self.metadata = metadata


def canonical_name(value: Any) -> str:
    name = str(value).strip().casefold()
    if not name or any(ord(c) < 32 for c in name):
        raise SchemaError("NOME_COLUNA_INVALIDO")
    return name


def identifier(value: str) -> str:
    """Identificadores vêm da configuração/catálogo, nunca de SQL livre."""
    if not isinstance(value, str) or not value.strip() or len(value) > 255:
        raise ValueError("Identificador vazio ou longo demais.")
    if any(ord(c) < 32 for c in value):
        raise ValueError("Identificador contém caracteres de controle.")
    return '"' + value.replace('"', '""') + '"'


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} deve ser inteiro positivo.")


@dataclass(frozen=True)
class RuleConfig:
    """Limiares do MVP, versionados e configuráveis por base. Não calibrados."""

    z_threshold: float = 3.0
    delta_threshold: float = 1.5
    min_evidences: int = 2
    version: str = "heuristica-z-v1"

    def __post_init__(self) -> None:
        for name in ("z_threshold", "delta_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} deve ser número finito positivo.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser número finito positivo.")
        _positive_int(self.min_evidences, "min_evidences")
        if not self.version.strip():
            raise ValueError("version não pode ser vazia.")


@dataclass(frozen=True)
class PartitionFilter:
    """Predicado restrito a coluna de partição: eq/gte/lt, sem expressão livre."""

    column: str
    value: str | int | float
    operator: str = "eq"

    def __post_init__(self) -> None:
        identifier(self.column)
        if self.operator not in ("eq", "gte", "lt"):
            raise ValueError("Operador permitido: eq, gte ou lt.")
        if isinstance(self.value, bool) or not isinstance(self.value, (str, int, float)):
            raise ValueError("Valor do filtro deve ser texto ou número.")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Filtro numérico deve ser finito.")
        if isinstance(self.value, str) and any(ord(c) < 32 for c in self.value):
            raise ValueError("Filtro contém caracteres de controle.")


@dataclass(frozen=True)
class SourceSpec:
    base: str
    table: str
    event_key: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()
    partition_filters: tuple[PartitionFilter, ...] = ()
    available_at_d0_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.base not in VALID_BASES:
            raise ValueError("base deve ser PF ou PJ.")
        identifier(self.table)
        for group in (self.event_key, self.order_by, self.available_at_d0_columns):
            if len({canonical_name(x) for x in group}) != len(group):
                raise ValueError("Colunas repetidas na configuração.")
            for name in group:
                identifier(name)


@dataclass(frozen=True)
class Config:
    region: str = "sa-east-1"
    database: str = "workspace_db"
    catalog: str = "AwsDataCatalog"
    workgroup: str = "primary"
    sources: tuple[SourceSpec, ...] = field(default_factory=lambda: (
        SourceSpec("PF", "AlertasInsider_PF_202608"),
        SourceSpec("PJ", "AlertasInsider_PJ_202608"),
    ))
    mode: str = "RETROSPECTIVO"
    rules: RuleConfig = field(default_factory=RuleConfig)
    rules_by_base: Mapping[str, RuleConfig] = field(default_factory=dict)
    max_rows: int = 10_000
    query_timeout_seconds: float = 180.0
    poll_seconds: float = 2.0
    max_scanned_bytes: int = 2 * 1024**3
    max_result_bytes: int = 64 * 1024**2
    max_dataframe_bytes: int = 256 * 1024**2
    require_workgroup_scan_limit: bool = False
    result_reuse_minutes: int = 60
    s3_output: str | None = None
    expected_bucket_owner: str | None = None
    encryption: str | None = None
    kms_key: str | None = None
    remove_exact_duplicates: bool = True
    numeric_locale: str = "strict"
    history_columns: tuple[str, ...] = ("resultado_final",)
    include_history: bool = False
    hmac_key_env: str = "ALERTAS_HMAC_KEY"

    def __post_init__(self) -> None:
        for name in ("database", "catalog", "workgroup"):
            identifier(getattr(self, name))
        if not self.region.strip() or not self.sources:
            raise ValueError("Região e fontes são obrigatórias.")
        if self.catalog != "AwsDataCatalog":
            raise ValueError("Este adaptador Glue usa AwsDataCatalog; catálogo federado requer outro adaptador.")
        if len({s.base for s in self.sources}) != len(self.sources):
            raise ValueError("Configure uma fonte por base; PF/PJ continuam separados.")
        if self.mode not in ("RETROSPECTIVO", "PROSPECTIVO"):
            raise ValueError("mode deve ser RETROSPECTIVO ou PROSPECTIVO.")
        if self.numeric_locale not in ("strict", "pt_BR"):
            raise ValueError("numeric_locale deve ser strict ou pt_BR.")
        for name in ("max_rows", "max_scanned_bytes", "max_result_bytes", "max_dataframe_bytes"):
            _positive_int(getattr(self, name), name)
        for name in ("require_workgroup_scan_limit", "remove_exact_duplicates", "include_history"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} deve ser booleano JSON, não texto.")
        for name in ("query_timeout_seconds", "poll_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser finito positivo.")
        if self.poll_seconds > 30:
            raise ValueError("poll_seconds deve ser <= 30.")
        if type(self.result_reuse_minutes) is not int or not 0 <= self.result_reuse_minutes <= 10_080:
            raise ValueError("result_reuse_minutes fora do intervalo permitido.")
        if self.mode == "PROSPECTIVO" and self.result_reuse_minutes != 0:
            raise ValueError("Modo prospectivo exige result_reuse_minutes=0 para evitar resultado antigo.")
        if self.encryption not in (None, "SSE_S3", "SSE_KMS"):
            raise ValueError("Use SSE_S3 ou SSE_KMS para resultados.")
        if self.encryption == "SSE_KMS" and not self.kms_key:
            raise ValueError("SSE_KMS exige kms_key.")
        if self.kms_key and self.encryption != "SSE_KMS":
            raise ValueError("kms_key exige encryption=SSE_KMS.")
        if self.s3_output and not re.fullmatch(r"s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/.*", self.s3_output):
            raise ValueError("s3_output deve ser s3://bucket/prefixo/.")
        if self.expected_bucket_owner and not re.fullmatch(r"\d{12}", self.expected_bucket_owner):
            raise ValueError("expected_bucket_owner deve ter 12 dígitos.")
        history = {canonical_name(x) for x in self.history_columns}
        if len(history) != len(self.history_columns) or not history:
            raise ValueError("history_columns deve ser não vazio, sem repetições.")
        if history.intersection(NUMERIC_FIELDS):
            raise ValueError("Coluna histórica não pode ser indicador de uma regra.")
        if set(self.rules_by_base) - VALID_BASES:
            raise ValueError("rules_by_base aceita apenas PF e PJ.")
        object.__setattr__(self, "rules_by_base", MappingProxyType(dict(self.rules_by_base)))
        active_rules = 3 if self.mode == "RETROSPECTIVO" else 1
        for source in self.sources:
            if self.rules_for(source.base).min_evidences > active_rules:
                raise ValueError("min_evidences excede o número de regras ativas neste modo.")
            routing = {canonical_name(x) for x in source.event_key + source.order_by + source.available_at_d0_columns}
            routing.update(canonical_name(x.column) for x in source.partition_filters)
            if history.intersection(routing):
                raise ValueError("Rótulos não podem determinar chave, ordenação ou filtro de entrada.")
            if self.mode == "PROSPECTIVO" and POST_EVENT_FIELDS.intersection(routing):
                raise ValueError("Indicador pós-evento não pode selecionar/identificar casos prospectivos.")

    def rules_for(self, base: str) -> RuleConfig:
        return self.rules_by_base.get(base, self.rules)

    def public_dict(self) -> dict[str, Any]:
        # O segredo HMAC nunca pertence ao objeto Config.
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        data["sources"] = [dataclasses.asdict(s) for s in self.sources]
        data["rules"] = dataclasses.asdict(self.rules)
        data["rules_by_base"] = {k: dataclasses.asdict(v) for k, v in self.rules_by_base.items()}
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Config:
        data = dict(values)
        if "sources" in data:
            sources = []
            for item in data["sources"]:
                source = dict(item)
                for name in ("event_key", "order_by", "available_at_d0_columns"):
                    if name in source:
                        source[name] = tuple(source[name])
                source["partition_filters"] = tuple(PartitionFilter(**x) for x in source.get("partition_filters", ()))
                sources.append(SourceSpec(**source))
            data["sources"] = tuple(sources)
        if "rules" in data:
            data["rules"] = RuleConfig(**data["rules"])
        if "rules_by_base" in data:
            data["rules_by_base"] = {k: RuleConfig(**v) for k, v in data["rules_by_base"].items()}
        if "history_columns" in data:
            data["history_columns"] = tuple(data["history_columns"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CatalogSchema:
    columns: Mapping[str, str]  # nome canônico -> nome exato do Glue
    types: Mapping[str, str]
    partitions: frozenset[str] = frozenset()

    @classmethod
    def from_glue(cls, table: Mapping[str, Any]) -> CatalogSchema:
        all_columns = table.get("StorageDescriptor", {}).get("Columns", []) + table.get("PartitionKeys", [])
        columns, types = {}, {}
        for col in all_columns:
            name = canonical_name(col["Name"])
            if name in columns:
                raise SchemaError("COLISOES_NO_CATALOGO")
            columns[name] = col["Name"]
            types[name] = str(col.get("Type", "unknown")).casefold()
        if not columns:
            raise SchemaError("CATALOGO_SEM_COLUNAS")
        return cls(columns, types, frozenset(canonical_name(x["Name"]) for x in table.get("PartitionKeys", [])))


@dataclass(frozen=True)
class QueryPlan:
    sql: str
    columns: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass
class FetchResult:
    data: pd.DataFrame
    source: SourceSpec
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertRepository(Protocol):
    def fetch(self, source: SourceSpec) -> FetchResult:
        """Retorna lote limitado com metadados de leitura."""
        ...


def _filter_literal(item: PartitionFilter, schema: CatalogSchema) -> str:
    name = canonical_name(item.column)
    kind = schema.types[name]
    value = item.value
    numeric_types = ("tinyint", "smallint", "int", "integer", "bigint", "float", "double", "decimal")
    if kind.startswith(numeric_types):
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(value)):
            raise ValueError("Partição numérica exige literal numérico.")
        return str(value)
    escaped = str(value).replace("'", "''")
    if kind == "date":
        date.fromisoformat(str(value))
        return f"DATE '{escaped}'"
    if kind not in ("string", "varchar", "char") and not kind.startswith(("varchar(", "char(")):
        raise ValueError("Tipo de partição não suportado; use adaptador tipado específico.")
    return f"'{escaped}'"


def build_query(config: Config, source: SourceSpec, schema: CatalogSchema) -> QueryPlan:
    """SELECT construído por catálogo. Não há parâmetro para SQL arbitrário.

    Lê todas as entradas do registro para comparar duplicatas conservadoramente.
    A seleção de colunas é explícita e registra o contrato, sem SELECT *.
    O rótulo é excluído por padrão. No modo prospectivo excluímos D1/variações.
    """
    history = {canonical_name(x) for x in config.history_columns}
    blocked = POST_EVENT_FIELDS if config.mode == "PROSPECTIVO" else frozenset()
    if config.mode == "PROSPECTIVO":
        allowed = {Z0, *(canonical_name(x) for x in source.event_key + source.available_at_d0_columns)}
        unknown = allowed - set(schema.columns)
        if unknown:
            raise SchemaError("COLUNA_PROSPECTIVA_AUSENTE")
        if config.include_history:
            allowed.update(history)
        fields = tuple(name for name in schema.columns if name in allowed and name not in blocked)
    else:
        fields = tuple(name for name in schema.columns if config.include_history or name not in history)
    if not fields or Z0 not in schema.columns:
        raise SchemaError("Z_D0_AUSENTE_NO_CATALOGO")
    for col in source.event_key + source.order_by:
        if canonical_name(col) not in fields:
            raise SchemaError("CHAVE_OU_ORDENACAO_AUSENTE")
    predicates = []
    for item in source.partition_filters:
        name = canonical_name(item.column)
        if name not in schema.partitions:
            raise ValueError("Filtros de entrada permitidos apenas em partições catalogadas.")
        op = {"eq": "=", "gte": ">=", "lt": "<"}[item.operator]
        predicates.append(f"{identifier(schema.columns[name])} {op} {_filter_literal(item, schema)}")
    selected = ", ".join(identifier(schema.columns[name]) for name in fields)
    sql = f"SELECT {selected} FROM {identifier(config.catalog)}.{identifier(config.database)}.{identifier(source.table)}"
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    if source.order_by:
        sql += " ORDER BY " + ", ".join(identifier(schema.columns[canonical_name(x)]) + " ASC NULLS LAST" for x in source.order_by)
    sql += f" LIMIT {config.max_rows + 1}"  # linha sentinela detecta recorte real
    warnings = []
    if not source.order_by:
        warnings.append("RECORTE_SEM_ORDENACAO_DETERMINISTICA")
    else:
        warnings.append("ORDENACAO_REQUER_CHAVE_UNICA_PARA_DESEMPATE")
    if config.mode == "RETROSPECTIVO" and Z1 not in schema.columns:
        warnings.append("Z_D1_AUSENTE_NO_CATALOGO")
    return QueryPlan(sql, tuple(schema.columns[x] for x in fields), {
        "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
        "schema_sha256": _digest({"types": dict(schema.types), "partitions": sorted(schema.partitions)}),
        "projection_complete_for_mode": config.mode == "RETROSPECTIVO",
        "warnings": warnings,
        "selection": "LIMIT_NAO_ALEATORIO",
        "partition_filter_count": len(source.partition_filters),
        "selected_column_count": len(fields),
    })


# -----------------------------------------------------------------------------
# 2. Adaptadores. Boto3 direto permite timeout/cancelamento e paginação explícitos.
# -----------------------------------------------------------------------------
class AthenaRepository:
    def __init__(self, config: Config, athena: Any, glue: Any, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.config, self.athena, self.glue = config, athena, glue
        self.clock, self.sleep = clock, sleep
        self._schemas: dict[str, CatalogSchema] = {}
        self._workgroup_limit: int | None = None
        self._workgroup_checked = False

    @classmethod
    def from_config(cls, config: Config, session: Any = None) -> AthenaRepository:
        # Credenciais pela role do SageMaker/cadeia do SDK; nenhuma chave no código.
        try:
            import boto3
            from botocore.config import Config as SDKConfig
        except ImportError as exc:
            raise RuntimeError("Instale boto3 para executar a fonte AWS.") from exc
        session = session or boto3.Session(region_name=config.region)
        sdk = SDKConfig(region_name=config.region, connect_timeout=5, read_timeout=20,
                        retries={"mode": "standard", "total_max_attempts": 4})
        return cls(config, session.client("athena", config=sdk), session.client("glue", config=sdk))

    def catalog(self, source: SourceSpec) -> CatalogSchema:
        if source.table not in self._schemas:
            response = self.glue.get_table(DatabaseName=self.config.database, Name=source.table)
            self._schemas[source.table] = CatalogSchema.from_glue(response["Table"])
        return self._schemas[source.table]

    def _check_workgroup(self) -> None:
        if self._workgroup_checked or not self.config.require_workgroup_scan_limit:
            return
        response = self.athena.get_work_group(WorkGroup=self.config.workgroup)
        limit = response["WorkGroup"].get("Configuration", {}).get("BytesScannedCutoffPerQuery")
        if not limit or limit > self.config.max_scanned_bytes:
            raise SourceError("WORKGROUP_SEM_LIMITE_COMPATIVEL")
        self._workgroup_limit = int(limit)
        self._workgroup_checked = True

    def fetch(self, source: SourceSpec) -> FetchResult:
        schema = self.catalog(source)
        plan = build_query(self.config, source, schema)
        self._check_workgroup()
        started = self.clock()
        deadline = started + self.config.query_timeout_seconds
        kwargs: dict[str, Any] = {
            "QueryString": plan.sql,
            "QueryExecutionContext": {"Database": self.config.database, "Catalog": self.config.catalog},
            "WorkGroup": self.config.workgroup,
            # Idempotência dos retries desta submissão; não reaproveitar entre novas execuções.
            "ClientRequestToken": uuid.uuid4().hex,
            "ResultReuseConfiguration": {"ResultReuseByAgeConfiguration": {
                "Enabled": self.config.result_reuse_minutes > 0,
                **({"MaxAgeInMinutes": self.config.result_reuse_minutes} if self.config.result_reuse_minutes else {}),
            }},
        }
        result_config: dict[str, Any] = {}
        if self.config.s3_output:
            result_config["OutputLocation"] = self.config.s3_output
        if self.config.expected_bucket_owner:
            result_config["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        if self.config.encryption:
            result_config["EncryptionConfiguration"] = {"EncryptionOption": self.config.encryption}
            if self.config.kms_key:
                result_config["EncryptionConfiguration"]["KmsKey"] = self.config.kms_key
        if result_config:
            kwargs["ResultConfiguration"] = result_config
        query_id, query_running = None, False
        metadata: dict[str, Any] = dict(plan.metadata)
        try:
            query_id = self.athena.start_query_execution(**kwargs)["QueryExecutionId"]
            query_running = True
            while True:
                if self.clock() >= deadline:
                    raise SourceError("QUERY_TIMEOUT", query_id)
                execution = self.athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
                status = execution["Status"]
                state = status["State"]
                stats = execution.get("Statistics", {})
                scanned = int(stats.get("DataScannedInBytes", 0))
                metadata.update(query_execution_id=query_id, query_state=state,
                                scanned_bytes=scanned,
                                engine_ms=stats.get("EngineExecutionTimeInMillis"),
                                total_query_ms=stats.get("TotalExecutionTimeInMillis"),
                                reused_result=bool(stats.get("ResultReuseInformation", {}).get("ReusedPreviousResult", False)),
                                effective_encryption=execution.get("ResultConfiguration", {}).get("EncryptionConfiguration", {}).get("EncryptionOption"))
                if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    query_running = False
                if scanned > self.config.max_scanned_bytes:
                    raise SourceError("QUERY_SCAN_BUDGET_EXCEEDED", query_id, scanned_bytes=scanned)
                if state == "SUCCEEDED":
                    break
                if state in ("FAILED", "CANCELLED"):
                    # A razão da AWS pode conter valores/SQL. Guardamos apenas seu hash.
                    reason = str(status.get("StateChangeReason", ""))
                    raise SourceError("QUERY_" + state, query_id,
                                      reason_sha256=hashlib.sha256(reason.encode()).hexdigest())
                if state not in ("QUEUED", "RUNNING"):
                    raise SourceError("QUERY_UNKNOWN_STATE", query_id)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise SourceError("QUERY_TIMEOUT", query_id)
                self.sleep(min(self.config.poll_seconds, remaining))
            rows: list[list[Any]] = []
            token, first_page, size_bytes = None, True, 0
            seen_tokens: set[str] = set()
            while True:
                if self.clock() >= deadline:
                    raise SourceError("RESULT_DOWNLOAD_TIMEOUT", query_id)
                page_args: dict[str, Any] = {"QueryExecutionId": query_id, "MaxResults": 1000}
                if token:
                    page_args["NextToken"] = token
                page = self.athena.get_query_results(**page_args)
                result = page["ResultSet"]
                columns = tuple(x["Name"] for x in result["ResultSetMetadata"]["ColumnInfo"])
                if columns != plan.columns:
                    raise SourceError("RESULT_SCHEMA_CHANGED", query_id)
                items = result.get("Rows", [])
                if first_page and items:
                    header = tuple(x.get("VarCharValue") for x in items[0].get("Data", []))
                    if header != columns:
                        raise SourceError("RESULT_HEADER_INVALID", query_id)
                    items = items[1:]
                first_page = False
                for row in items:
                    values = [cell.get("VarCharValue") for cell in row.get("Data", [])]
                    if len(values) != len(columns):
                        raise SourceError("RESULT_ROW_SHAPE_INVALID", query_id)
                    size_bytes += sum(len(x.encode("utf-8")) for x in values if x is not None)
                    if size_bytes > self.config.max_result_bytes:
                        raise SourceError("RESULT_MEMORY_BUDGET_EXCEEDED", query_id)
                    rows.append(values)
                    if len(rows) > self.config.max_rows + 1:
                        raise SourceError("RESULT_ROW_LIMIT_VIOLATED", query_id)
                token = page.get("NextToken")
                if not token:
                    break
                if token in seen_tokens:
                    raise SourceError("RESULT_PAGINATION_LOOP", query_id)
                seen_tokens.add(token)
            if self.clock() >= deadline:
                raise SourceError("RESULT_DOWNLOAD_TIMEOUT", query_id)
            frame = pd.DataFrame(rows[:self.config.max_rows], columns=plan.columns)
            memory = int(frame.memory_usage(index=True, deep=True).sum())
            if memory > self.config.max_dataframe_bytes:
                raise SourceError("DATAFRAME_MEMORY_BUDGET_EXCEEDED", query_id)
            warnings = list(metadata["warnings"])
            if self._workgroup_limit is None:
                warnings.append("LIMITE_BYTES_APLICACAO_BEST_EFFORT_USE_WORKGROUP_PARA_LIMITE_SERVIDOR")
            if metadata.get("reused_result"):
                warnings.append("RESULTADO_REUTILIZADO_PODE_ESTAR_DESATUALIZADO")
            metadata.update(warnings=warnings, workgroup=self.config.workgroup,
                            workgroup_scan_limit=self._workgroup_limit,
                            rows_returned_with_sentinel=len(rows),
                            truncated=len(rows) > self.config.max_rows,
                            dataframe_bytes=memory, elapsed_seconds=round(self.clock() - started, 3))
            return FetchResult(frame, source, metadata)
        except BaseException as exc:
            cancel_requested, cancel_error = False, None
            if query_id and query_running:
                try:
                    self.athena.stop_query_execution(QueryExecutionId=query_id)
                    cancel_requested = True
                except Exception as cancel_exc:
                    cancel_error = type(cancel_exc).__name__
            if isinstance(exc, SourceError):
                exc.metadata.update(cancel_requested=cancel_requested, cancel_error=cancel_error)
                if "scanned_bytes" in metadata:
                    exc.metadata.setdefault("scanned_bytes", metadata["scanned_bytes"])
                raise
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise SourceError("AWS_REQUEST_FAILED", query_id, error_type=type(exc).__name__,
                              cancel_requested=cancel_requested, cancel_error=cancel_error) from exc


class FrameRepository:
    """Adaptador offline para testes e notebooks. Não acessa AWS."""

    def __init__(self, config: Config, frames: Mapping[str, pd.DataFrame]):
        self.config, self.frames = config, frames

    def fetch(self, source: SourceSpec) -> FetchResult:
        raw = self.frames[source.base]
        data = raw.iloc[:self.config.max_rows].copy()
        # Ordenação/filtro não são simulados aqui: impedir falsa equivalência à query.
        if source.order_by or source.partition_filters:
            raise ValueError("FrameRepository exige lote previamente selecionado e SourceSpec sem filtros/ordenação.")
        return FetchResult(data, source, {
            "selection": "LOTE_OFFLINE_FORNECIDO",
            "projection_complete_for_mode": self.config.mode == "RETROSPECTIVO",
            "truncated": len(raw) > self.config.max_rows,
            "rows_returned_with_sentinel": min(len(raw), self.config.max_rows + 1),
            "warnings": [],
        })


# -----------------------------------------------------------------------------
# 3. Qualidade. Deduplicação das entradas brutas ANTES de coerção numérica.
# -----------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (dict, list, tuple, set, bytes)):
        return False
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _token(value: Any) -> Any:
    """Comparação exata tipada, inclusive conteúdo aninhado; não por indicadores."""
    if _is_missing(value):
        return ["null"]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, (datetime, date)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_token(x) for x in value]]
    if isinstance(value, dict):
        pairs = [[_token(k), _token(v)] for k, v in value.items()]
        return ["dict", sorted(pairs, key=_json)]
    raise SchemaError("TIPO_BRUTO_NAO_SUPORTADO")


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    names = [canonical_name(c) for c in raw.columns]
    if len(set(names)) != len(names):
        raise SchemaError("COLUNAS_COLIDEM_APOS_NORMALIZACAO")
    result = raw.copy(deep=True).reset_index(drop=True)
    result.columns = names
    return result


def _number_column(series: pd.Series, locale: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Retorna número, ausente, inválido. Não transforma erro em zero.

    strict: 1.23/1e3; pt_BR: 1.234,56. Valores já numéricos não são reinterpretados.
    Strings com %, moedas e formatos ambíguos são rejeitadas.
    """
    missing = series.map(lambda x: _is_missing(x) or (isinstance(x, str) and not x.strip()))
    candidate = series.astype("object").copy()
    invalid_type = series.map(lambda x: isinstance(x, (bool, np.bool_)) or
                             (not _is_missing(x) and not isinstance(x, (str, int, float, np.integer, np.floating, Decimal))))
    if locale == "pt_BR":
        pattern = re.compile(r"[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?:[eE][+-]?\d+)?\Z")
        for index, value in series.items():
            if isinstance(value, str) and not missing.loc[index]:
                text = value.strip()
                if not pattern.fullmatch(text):
                    invalid_type.loc[index] = True
                else:
                    candidate.loc[index] = text.replace(".", "").replace(",", ".")
    candidate.loc[missing | invalid_type] = None
    numeric = pd.to_numeric(candidate, errors="coerce").astype("float64")
    finite = pd.Series(np.isfinite(numeric.to_numpy()), index=series.index)
    invalid = (~missing) & (invalid_type | ~finite)
    numeric.loc[~finite | invalid] = np.nan
    return numeric, missing.astype(bool), invalid.astype(bool)


@dataclass
class PreparedBatch:
    source: SourceSpec
    features: pd.DataFrame
    issues: list[list[dict[str, str]]]
    history: pd.DataFrame
    graph_keys: dict[str, dict[str, Any]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class GraphContext:
    """Contrato agregado: sem nomes/documentos/texto livre ou alteração de score."""

    related_events: int = 0
    related_entities: int = 0

    def __post_init__(self) -> None:
        for value in (self.related_events, self.related_entities):
            if type(value) is not int or value < 0:
                raise ValueError("Contagens do grafo devem ser inteiros não negativos.")


class GraphRepository(Protocol):
    """Porta de leitura para Neptune; o adapter real depende do schema do grafo.

    lookup_keys contém a chave de evento verificada dentro do processo. O adapter
    deve usar consultas parametrizadas, timeout e credenciais de leitura.
    Valores brutos nunca devem ir para logs, LLM ou relatórios.
    """

    def read_context(self, base: str, lookup_keys: Mapping[str, Mapping[str, Any]]) -> Mapping[str, GraphContext]:
        ...


class NullGraphRepository:
    """Execução válida sem Neptune, sem inventar relações."""

    def read_context(self, base: str, lookup_keys: Mapping[str, Mapping[str, Any]]) -> Mapping[str, GraphContext]:
        return {}


@dataclass
class BaseResult:
    cases: pd.DataFrame
    history: pd.DataFrame  # canal separado: nunca enviado ao motor de regras
    audit: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "base": self.audit["base"],
            "rows_read": self.audit["rows_read"],
            "exact_duplicates_removed": self.audit["exact_duplicates_removed"],
            "cases": len(self.cases),
            "priority_counts": {str(k): int(v) for k, v in self.cases["prioridade_exploratoria"].value_counts().items()},
            "data_status_counts": {str(k): int(v) for k, v in self.cases["status_dados"].value_counts().items()},
            "truncated": self.audit["source"].get("truncated", False),
        }

    def review_view(self, limit: int = 20) -> pd.DataFrame:
        """Só dados canônicos e IDs pseudônimos; nenhum identificador da origem."""
        _positive_int(limit, "limit")
        cols = ["base", "case_id", "prioridade_exploratoria", "status_dados",
                "quantidade_evidencias", "cobertura_regras", "evidencias", "problemas_dados"]
        return self.cases[cols].head(limit).copy(deep=True)


class AlertPipeline:
    def __init__(self, config: Config, *, graph: GraphRepository | None = None,
                 hmac_key: bytes | None = None, run_id: str | None = None):
        self.config = config
        self.graph = graph or NullGraphRepository()
        env_key = os.environ.get(config.hmac_key_env)
        key = hmac_key or (env_key.encode("utf-8") if env_key else secrets.token_bytes(32))
        if len(key) < 32:
            raise ValueError("A chave HMAC deve ter pelo menos 32 bytes.")
        self._key = key
        self.identity_scope = "PERSISTENTE_POR_CHAVE_HMAC" if hmac_key or env_key else "APENAS_ESTA_EXECUCAO"
        self.run_id = run_id or uuid.uuid4().hex

    def prepare(self, fetched: FetchResult) -> PreparedBatch:
        """Contrato canônico + qualidade, com o histórico isolado na entrada."""
        config, source = self.config, fetched.source
        raw = _normalize_columns(fetched.data)
        if len(raw) > config.max_rows:
            raise SchemaError("LOTE_EXCEDE_MAX_ROWS")
        if int(raw.memory_usage(index=True, deep=True).sum()) > config.max_dataframe_bytes:
            raise SchemaError("LOTE_EXCEDE_ORCAMENTO_MEMORIA")
        if Z0 not in raw.columns:
            raise SchemaError("Z_D0_AUSENTE")
        event_columns = tuple(canonical_name(x) for x in source.event_key)
        if set(event_columns) - set(raw.columns):
            raise SchemaError("CHAVE_EVENTO_AUSENTE")
        history_columns = [canonical_name(x) for x in config.history_columns if canonical_name(x) in raw.columns]
        history_raw = raw[history_columns].copy()
        raw = raw.drop(columns=history_columns)
        # Nem a identidade nem a deduplicação usam o rótulo histórico.
        # Prospectivo também não usa pós-evento para identidade/deduplicação.
        if config.mode == "PROSPECTIVO":
            allowed = {Z0, *(canonical_name(x) for x in source.event_key + source.available_at_d0_columns)}
            raw = raw[[c for c in raw.columns if c in allowed]].copy()
        ordered_columns = sorted(raw.columns)
        signatures = pd.Series([
            _json([[name, _token(value)] for name, value in zip(ordered_columns, row)])
            for row in raw[ordered_columns].itertuples(index=False, name=None)
        ], index=raw.index, dtype="object")
        complete_inputs = bool(fetched.metadata.get("projection_complete_for_mode", False)) and config.mode == "RETROSPECTIVO"
        if not complete_inputs and not event_columns:
            # Indicadores iguais não identificam um evento. Preservar cada ocorrência.
            id_inputs = pd.Series([self.run_id + "\0" + str(i) + "\0" + text for i, text in enumerate(signatures)], index=raw.index, dtype="object")
            identity_scope = "POR_POSICAO_NESTA_EXECUCAO_SEM_CHAVE_VERIFICADA"
        else:
            id_inputs, identity_scope = signatures, self.identity_scope
        namespace = "\0".join((config.catalog, config.database, source.table, source.base)) + "\0"
        ids = id_inputs.map(lambda text: hmac.new(self._key, (namespace + text).encode("utf-8"), hashlib.sha256).hexdigest())
        duplicate_counts = signatures.map(signatures.value_counts()).astype("int64")
        removed = signatures.duplicated(keep="first")
        # Não remover duplicatas de uma projeção incompleta recebida de outro adapter.
        can_remove = config.remove_exact_duplicates and complete_inputs
        keep = ~removed if can_remove else pd.Series(True, index=raw.index)
        if not config.include_history:
            history_raw = pd.DataFrame(index=raw.index)
        history = history_raw.copy()
        history.insert(0, "case_id", ids)
        history.insert(0, "base", source.base)
        history.insert(2, "source_position", np.arange(len(raw), dtype="int64"))
        history = history.reset_index(drop=True)
        features = pd.DataFrame({
            "base": source.base,
            "case_id": ids.loc[keep],
            "source_position": np.arange(len(raw), dtype="int64")[keep.to_numpy()],
            "ocorrencias_entradas_identicas": duplicate_counts.loc[keep],
        }).reset_index(drop=True)
        raw_kept = raw.loc[keep].reset_index(drop=True)
        issues: list[list[dict[str, str]]] = [[] for _ in range(len(raw_kept))]
        quality_by_column = {}
        active_fields = (Z0, Z1) if config.mode == "RETROSPECTIVO" else (Z0,)
        fields = NUMERIC_FIELDS if config.mode == "RETROSPECTIVO" else (Z0,)
        for col in fields:
            present = col in raw_kept.columns
            values = raw_kept[col] if present else pd.Series(None, index=raw_kept.index, dtype="object")
            numeric, missing, invalid = _number_column(values, config.numeric_locale)
            features[col] = numeric
            quality_by_column[col] = {"present": present, "missing": int(missing.sum()), "invalid": int(invalid.sum())}
            if col in active_fields or present:
                for i in np.flatnonzero((missing | invalid).to_numpy()):
                    code = "NUMERO_INVALIDO" if invalid.iloc[i] else ("VALOR_AUSENTE" if present else "COLUNA_AUSENTE")
                    issues[int(i)].append({"campo": col, "codigo": code})
        with np.errstate(over="ignore", invalid="ignore"):
            features["delta_z"] = features[Z1] - features[Z0] if config.mode == "RETROSPECTIVO" else np.nan
        overflow = features["delta_z"].notna() & ~np.isfinite(features["delta_z"])
        features.loc[overflow, "delta_z"] = np.nan
        for i in np.flatnonzero(overflow.to_numpy()):
            issues[int(i)].append({"campo": "delta_z", "codigo": "NUMERO_INVALIDO"})
        features["chave_evento_repetida"] = False
        features["ocorrencias_chave_evento"] = 1
        graph_keys: dict[str, dict[str, Any]] = {}
        if event_columns:
            valid_key = pd.Series(True, index=raw_kept.index)
            for col in event_columns:
                missing = raw_kept[col].map(lambda x: _is_missing(x) or (isinstance(x, str) and not x.strip()))
                valid_key &= ~missing
                for i in np.flatnonzero(missing.to_numpy()):
                    issues[int(i)].append({"campo": col, "codigo": "CHAVE_EVENTO_INCOMPLETA"})
            key_signatures = raw_kept.loc[valid_key, list(event_columns)].apply(
                lambda row: _json([_token(x) for x in row]), axis=1) if valid_key.any() else pd.Series(dtype="object")
            counts = key_signatures.map(key_signatures.value_counts())
            features.loc[valid_key, "ocorrencias_chave_evento"] = counts.astype("int64")
            features.loc[valid_key, "chave_evento_repetida"] = counts.gt(1).to_numpy()
            for i in np.flatnonzero(valid_key.to_numpy()):
                graph_keys[str(features.iloc[i]["case_id"])] = {c: raw_kept.iloc[i][c] for c in event_columns}
        warnings = list(fetched.metadata.get("warnings", []))
        if not event_columns:
            warnings.append("CHAVE_NEGOCIO_NAO_VALIDADA_REPETICOES_NAO_SAO_CONSOLIDADAS")
        if not complete_inputs:
            warnings.append("PROJECAO_INCOMPLETA_DUPLICATAS_APENAS_SINALIZADAS")
        audit = {
            "run_id": self.run_id, "base": source.base, "table": source.table,
            "version": VERSION, "mode": config.mode,
            "rules": dataclasses.asdict(config.rules_for(source.base)),
            "rows_read": len(raw), "rows_after_dedup": len(features),
            "exact_duplicates_detected": int(removed.sum()),
            "exact_duplicates_removed": int(removed.sum()) if can_remove else 0,
            "dedup_scope": "TODAS_ENTRADAS_BRUTAS_SEM_ROTULOS" if complete_inputs else "PROJECAO_APENAS_SINALIZADA",
            "quality_by_column": quality_by_column,
            "history_columns_isolated": history_columns if config.include_history else [],
            "identity_scope": identity_scope,
            "config_sha256": _digest(config.public_dict()),
            "source": dict(fetched.metadata), "warnings": warnings,
        }
        return PreparedBatch(source, features, issues, history, graph_keys, audit)

    def score(self, batch: PreparedBatch) -> pd.DataFrame:
        """Só features e diagnóstico. Não há argumento para resultado_final."""
        df = batch.features.copy(deep=True)
        rules = self.config.rules_for(batch.source.base)
        n = len(df)
        evidence: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        available_count = np.zeros(n, dtype="int64")
        hits = np.zeros(n, dtype="int64")
        rule_specs = [("Z_D0_EXTREMO", Z0, rules.z_threshold)]
        if self.config.mode == "RETROSPECTIVO":
            rule_specs += [("Z_D1_EXTREMO", Z1, rules.z_threshold), ("DELTA_Z_RELEVANTE", "delta_z", rules.delta_threshold)]
        for code, col, threshold in rule_specs:
            available = df[col].notna().to_numpy()
            mask = available & df[col].abs().ge(threshold).to_numpy()
            available_count += available.astype("int64")
            hits += mask.astype("int64")
            for i in np.flatnonzero(mask):
                value = float(df.iloc[i][col])
                evidence[int(i)].append({
                    "codigo": code, "campo": col, "valor": value,
                    "operador": "abs(valor) >= limiar", "limiar": float(threshold),
                    "familia": "Z_SCORE", "versao_regra": rules.version,
                    "descricao": f"{col}: |{value:.4g}| >= {threshold:g}",
                })
        active_fields = (Z0, Z1) if self.config.mode == "RETROSPECTIVO" else (Z0,)
        valid_fields = df[list(active_fields)].notna().sum(axis=1).to_numpy()
        invalid_rows = np.array([any(p["codigo"] == "NUMERO_INVALIDO" and p["campo"] in (*active_fields, "delta_z") for p in items)
                                 for items in batch.issues], dtype=bool)
        incomplete_key = np.array([any(p["codigo"] == "CHAVE_EVENTO_INCOMPLETA" for p in items)
                                   for items in batch.issues], dtype=bool)
        quality = np.full(n, "VALIDO", dtype=object)
        quality[(valid_fields < len(active_fields)) | incomplete_key] = "INCOMPLETO"
        quality[valid_fields == 0] = "SEM_DADOS"
        quality[invalid_rows] = "INVALIDO"
        priority = np.where(hits >= rules.min_evidences, "REVISAR", "OBSERVAR").astype(object)
        priority[quality != "VALIDO"] = "VALIDAR_DADOS"
        df["status_dados"] = quality
        df["prioridade_exploratoria"] = priority
        df["quantidade_evidencias"] = hits
        df["regras_avaliaveis"] = available_count
        df["regras_ativas"] = len(rule_specs)
        df["cobertura_regras"] = available_count / len(rule_specs)
        df["evidencias"] = evidence
        df["problemas_dados"] = batch.issues
        df["status_revisao"] = "AGUARDANDO_REVISAO_HUMANA"
        return df

    def enrich(self, cases: pd.DataFrame, batch: PreparedBatch) -> pd.DataFrame:
        """O grafo acrescenta contexto; nunca altera evidências/regras/prioridade."""
        output = cases.copy(deep=True)
        contexts: Mapping[str, GraphContext] = {}
        status = "NAO_CONFIGURADO" if isinstance(self.graph, NullGraphRepository) else "SEM_CHAVE_EVENTO"
        if not isinstance(self.graph, NullGraphRepository) and batch.graph_keys:
            try:
                contexts = self.graph.read_context(batch.source.base, batch.graph_keys)
                if set(contexts) - set(output["case_id"]) or any(not isinstance(v, GraphContext) for v in contexts.values()):
                    raise ValueError("Resposta do grafo fora do contrato.")
                status = "CONSULTADO"
            except Exception as exc:
                contexts = {}
                status = "INDISPONIVEL"
                batch.audit["warnings"].append("GRAFO_INDISPONIVEL_CONTINUA_SEM_CONTEXTO")
                batch.audit["graph_error_type"] = type(exc).__name__
        output["contexto_grafo"] = [
            {"status": "ENRIQUECIDO", **dataclasses.asdict(contexts[x])} if x in contexts else {"status": status}
            for x in output["case_id"]
        ]
        batch.audit["graph_status"] = status
        batch.audit["graph_enriched_cases"] = sum(x in contexts for x in output["case_id"])
        return output

    def finalize(self, cases: pd.DataFrame, batch: PreparedBatch) -> BaseResult:
        output = cases.copy(deep=True)
        # VALIDAR_DADOS é uma fila de qualidade, não uma classe de risco.
        output["_queue"] = output["prioridade_exploratoria"].map({"VALIDAR_DADOS": 0, "REVISAR": 1, "OBSERVAR": 2})
        output["_magnitude"] = output[[c for c in (Z0, Z1) if c in output]].abs().max(axis=1)
        output = output.sort_values(["_queue", "quantidade_evidencias", "_magnitude", "case_id", "source_position"],
                                    ascending=[True, False, False, True, True], kind="stable")
        output = output.drop(columns=["_queue", "_magnitude"]).reset_index(drop=True)
        audit = dict(batch.audit)
        audit["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        audit["no_automatic_decisions"] = True
        audit["rules_not_calibrated"] = True
        audit["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result = BaseResult(output, batch.history, audit)
        audit["summary"] = result.summary()
        return result

    def run(self, fetched: FetchResult) -> BaseResult:
        batch = self.prepare(fetched)
        cases = self.score(batch)
        cases = self.enrich(cases, batch)
        return self.finalize(cases, batch)


# -----------------------------------------------------------------------------
# 5. Orquestração por base, auditoria e exportação opcional do dossiê canônico.
# -----------------------------------------------------------------------------
@dataclass
class RunReport:
    run_id: str
    results: dict[str, BaseResult]
    failures: dict[str, dict[str, Any]]

    @property
    def status(self) -> str:
        if self.results and not self.failures:
            return "SUCESSO"
        return "PARCIAL" if self.results else "FALHA"

    def summary(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "status": self.status,
                "bases": {k: v.summary() for k, v in self.results.items()}, "failures": self.failures}


def run_all(config: Config, repository: AlertRepository, *,
            pipeline: AlertPipeline | None = None) -> RunReport:
    pipeline = pipeline or AlertPipeline(config)
    if pipeline.config != config:
        raise ValueError("Pipeline e repository devem usar a mesma configuração.")
    if hasattr(repository, "config") and repository.config != config:
        raise ValueError("Repository e execução devem usar a mesma configuração.")
    results: dict[str, BaseResult] = {}
    failures: dict[str, dict[str, Any]] = {}
    for source in config.sources:
        try:
            fetched = repository.fetch(source)
            if fetched.source != source:
                raise SchemaError("FONTE_RETORNADA_DIVERGE_DA_SOLICITADA")
            result = pipeline.run(fetched)
            results[source.base] = result
            LOGGER.info("base=%s status=SUCESSO rows=%s cases=%s duplicates_removed=%s",
                        source.base, result.audit["rows_read"], len(result.cases), result.audit["exact_duplicates_removed"])
        except Exception as exc:
            failure: dict[str, Any] = {"type": type(exc).__name__, "code": "PIPELINE_FAILED"}
            if isinstance(exc, SourceError):
                failure.update(code=exc.code, query_execution_id=exc.query_id, **exc.metadata)
            elif isinstance(exc, SchemaError):
                failure["code"] = str(exc)
            failures[source.base] = failure
            # Não usar exc_info=True ou str(exc) para exceções externas: podem conter PII/SQL.
            LOGGER.error("base=%s status=FALHA type=%s code=%s", source.base, failure["type"], failure["code"])
    return RunReport(pipeline.run_id, results, failures)


def _atomic_text(path: Path, contents: str) -> None:
    # Arquivos de revisão têm dados operacionais. Criar com acesso só ao usuário.
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(contents)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def export_report(report: RunReport, root: str | Path, *, include_cases: bool = False) -> Path:
    """Padrão: só agregados/auditoria. Casos exigem include_cases=True explícito.

    O histórico humano nunca é exportado por esta função. Dados brutos, chaves de
    negócio e segredos não pertencem ao dossiê. HMAC é pseudonimização, não anonimização.
    """
    directory = Path(root) / report.run_id
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    _atomic_text(directory / "resumo.json", json.dumps(report.summary(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for base, result in report.results.items():
        _atomic_text(directory / f"auditoria_{base}.json", json.dumps(result.audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if include_cases:
            # JSONL evita CSV com células executáveis e preserva evidências estruturadas.
            _atomic_text(directory / f"casos_{base}.jsonl", result.cases.to_json(orient="records", lines=True, force_ascii=False, double_precision=15))
    return directory


# -----------------------------------------------------------------------------
# 6. Avaliação offline. Rótulos desconhecidos/nulos não são negativos.
# -----------------------------------------------------------------------------
def evaluate_offline(result: BaseResult, *, positive_labels: Sequence[str],
                     negative_labels: Sequence[str], label_column: str = "resultado_final") -> dict[str, Any]:
    """Confusão entre fila REVISAR e rótulo humano mapeado explicitamente.

    Não mede causalidade, ilícito nem desempenho de produção. Só o lote fornecido.
    Casos com dados insuficientes, rótulos indefinidos ou decisões conflitantes nas
    mesmas entradas são excluídos e contabilizados. Não treina nem calibra regras.
    """
    if isinstance(positive_labels, str) or isinstance(negative_labels, str):
        raise ValueError("Informe listas de rótulos, não uma string isolada.")
    if any(not isinstance(x, str) for x in (*positive_labels, *negative_labels)):
        raise ValueError("Mapeamentos de rótulos devem conter textos.")
    norm = lambda x: str(x).strip().casefold()
    positive, negative = {norm(x) for x in positive_labels}, {norm(x) for x in negative_labels}
    if not positive or not negative or positive & negative or "" in positive | negative:
        raise ValueError("Mapeamentos positivos/negativos devem ser explícitos, não vazios e disjuntos.")
    column = canonical_name(label_column)
    if column not in result.history:
        raise ValueError("Histórico não carregado. Use include_history=True para avaliação offline.")
    labels = result.history[["case_id", column]].copy()
    labels["normalized"] = labels[column].map(lambda x: "" if _is_missing(x) else norm(x))
    labels["truth"] = labels["normalized"].map({**{x: True for x in positive}, **{x: False for x in negative}})
    known = labels.dropna(subset=["truth"])
    unique_class_count = known.groupby("case_id")["truth"].nunique()
    conflicts = set(unique_class_count[unique_class_count > 1].index)
    truth_map = known.drop_duplicates("case_id").set_index("case_id")["truth"]
    cases = result.cases.drop_duplicates("case_id").copy()
    cases["truth"] = cases["case_id"].map(truth_map)
    conflict_mask = cases["case_id"].isin(conflicts)
    known_mask = cases["truth"].notna()
    quality_mask = cases["status_dados"].eq("VALIDO")
    eligible = cases.loc[known_mask & quality_mask & ~conflict_mask]
    actual = eligible["truth"].astype(bool)
    predicted = eligible["prioridade_exploratoria"].eq("REVISAR")
    tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
    fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
    divide = lambda num, den: num / den if den else None
    return {
        "base": result.audit["base"], "scope": "LOTE_OFFLINE_NAO_REPRESENTATIVO",
        "mode": result.audit["mode"], "evaluated_cases": len(eligible),
        "unique_cases": len(cases), "unknown_label_rows": int(labels["truth"].isna().sum()),
        "excluded_conflicting_labels": int(conflict_mask.sum()),
        "excluded_missing_or_unknown_label": int((~known_mask & ~conflict_mask).sum()),
        "excluded_data_quality": int((known_mask & ~quality_mask & ~conflict_mask).sum()),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "review_fraction": divide(tp + fp, len(eligible)),
        "positive_fraction": divide(tp + fn, len(eligible)),
        "rules": result.audit["rules"],
    }


# -----------------------------------------------------------------------------
# 7. Dados sintéticos e CLI: a demonstração nunca cria cliente AWS.
# -----------------------------------------------------------------------------
def demo_frames() -> dict[str, pd.DataFrame]:
    raw = pd.DataFrame([
        {"evento_demo": "A", Z0: "4.1", Z1: "5.9", PRICE: "0.02", CUSTODY: "0.03", "resultado_final": "PROCEDENTE"},
        {"evento_demo": "A", Z0: "4.1", Z1: "5.9", PRICE: "0.02", CUSTODY: "0.03", "resultado_final": "PROCEDENTE"},
        {"evento_demo": "B", Z0: "0.3", Z1: "0.4", PRICE: "0.01", CUSTODY: "0.01", "resultado_final": "IMPROCEDENTE"},
        {"evento_demo": "C", Z0: "texto inválido", Z1: "4.2", PRICE: None, CUSTODY: None, "resultado_final": None},
        {"evento_demo": "D", Z0: "-3.2", Z1: "-3.4", PRICE: None, CUSTODY: None, "resultado_final": "EM ANALISE"},
        {"evento_demo": "E", Z0: "1.1", Z1: None, PRICE: None, CUSTODY: None, "resultado_final": "IMPROCEDENTE"},
        # Mesmo indicador, outro evento: NÃO remover por igualdade de métricas.
        {"evento_demo": "F", Z0: "0.3", Z1: "0.4", PRICE: "0.01", CUSTODY: "0.01", "resultado_final": None},
    ])
    return {"PF": raw.copy(deep=True), "PJ": raw.copy(deep=True)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="Executa dados sintéticos sem AWS.")
    group.add_argument("--aws", action="store_true", help="Executa SELECT no Athena/Glue usando sua role.")
    parser.add_argument("--config", type=Path, help="JSON de configuração; nenhum segredo deve estar nele.")
    parser.add_argument("--output", type=Path, help="Exporta agregados/auditoria para diretório novo por execução.")
    parser.add_argument("--export-cases", action="store_true", help="Inclui JSONL canônico para revisão humana.")
    parser.add_argument("--include-history", action="store_true", help="Carrega rótulos apenas em canal offline separado.")
    args = parser.parse_args(argv)
    if args.export_cases and not args.output:
        parser.error("--export-cases exige --output.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = Config.from_json(args.config) if args.config else Config()
        if args.include_history:
            config = dataclasses.replace(config, include_history=True)
        repository: AlertRepository = FrameRepository(config, demo_frames()) if args.demo else AthenaRepository.from_config(config)
        report = run_all(config, repository)
        print(json.dumps(report.summary(), ensure_ascii=False, indent=2, allow_nan=False))
        if args.output:
            directory = export_report(report, args.output, include_cases=args.export_cases)
            print(f"Relatório gerado: {directory}")
        return 0 if report.status == "SUCESSO" else 2
    except Exception as exc:
        LOGGER.error("execucao_interrompida type=%s; confira configuração e dependências", type(exc).__name__)
        return 2


if __name__ == "__main__":
    sys.exit(main())
