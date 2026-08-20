"""POC DSPy 100% local, deterministica e sem chaves de API.

Compatibilidade validada com Python 3.12 e dspy/dspy-ai 3.3.0.
Depois da instalacao do pacote, este script nao usa rede nem baixa modelos.
"""

from __future__ import annotations

import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Literal

# Evita que o cache tente escrever em uma pasta global sem permissao.
os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).parent / ".dspy_cache"))

import dspy
from dspy.utils import DummyLM


Intent = Literal["cancelamento", "cobranca", "acesso", "informacao", "elogio"]
Sentiment = Literal["positivo", "negativo", "neutro"]


class ClassificarSuporte(dspy.Signature):
    """Classifique uma mensagem de suporte em uma intencao e um sentimento validos."""

    texto: str = dspy.InputField(desc="Mensagem enviada pelo cliente")
    intencao: Intent = dspy.OutputField(
        desc="Uma de: cancelamento, cobranca, acesso, informacao, elogio"
    )
    sentimento: Sentiment = dspy.OutputField(
        desc="Um de: positivo, negativo, neutro"
    )


class ClassificadorSuporte(dspy.Module):
    """Programa DSPy: a estrutura e fixa; as demonstracoes podem ser compiladas."""

    def __init__(self) -> None:
        super().__init__()
        self.classificar = dspy.ChainOfThought(ClassificarSuporte)

    def forward(self, texto: str) -> dspy.Prediction:
        return self.classificar(texto=texto)


def normalizar(texto: str) -> str:
    """Minusculas e remocao de acentos, apenas para o mock local."""
    decomposed = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def regras_locais(texto: str) -> tuple[Intent, Sentiment]:
    """Simula um modelo ultraleve; nao representa inteligencia linguistica real."""
    value = normalizar(texto)

    if any(word in value for word in ("cancel", "encerrar", "desistir")):
        intencao: Intent = "cancelamento"
    elif any(word in value for word in ("cobr", "fatura", "pagamento", "vencimento")):
        intencao = "cobranca"
    elif any(word in value for word in ("senha", "entrar", "login", "bloquead", "acesso")):
        intencao = "acesso"
    elif any(word in value for word in ("obrigad", "excelente", "parabens", "resolvido")):
        intencao = "elogio"
    else:
        intencao = "informacao"

    if any(
        word in value
        for word in ("irritad", "absurdo", "problema", "nao consigo", "duplic", "cancel")
    ):
        sentimento: Sentiment = "negativo"
    elif any(word in value for word in ("obrigad", "excelente", "parabens", "resolvido")):
        sentimento = "positivo"
    else:
        sentimento = "neutro"

    return intencao, sentimento


def extrair_texto_do_prompt(final_message: str) -> str:
    """Extrai somente o InputField; ignora a lista de rotulos do adaptador."""
    match = re.search(
        r"\[\[ ## texto ## \]\]\s*(.*?)(?:\n\n|\Z)",
        final_message,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else final_message


class DemoAwareDummyLM(DummyLM):
    """DummyLM deterministico que reage a demonstracoes inseridas pelo DSPy.

    Sem exemplos compilados, devolve uma classificacao padrao fraca. Quando o
    prompt contem demonstracoes, aplica as regras locais. Isso permite observar
    o efeito mecanico do BootstrapFewShot sem API, servidor ou modelo externo.
    """

    def __init__(self) -> None:
        # DummyLM nativo em modo dicionario; a resposta e definida em forward().
        super().__init__(answers={})

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt or ""}]
        final_message = messages[-1].get("content", "")

        # Demos DSPy aparecem como turnos assistant antes da pergunta final.
        possui_demos = any(
            message.get("role") == "assistant" for message in messages[:-1]
        )

        if possui_demos:
            texto = extrair_texto_do_prompt(final_message)
            intencao, sentimento = regras_locais(texto)
            resposta = {
                "reasoning": "Ha demonstracoes compiladas; aplicar regras locais deterministicas.",
                "intencao": intencao,
                "sentimento": sentimento,
            }
        else:
            resposta = {
                "reasoning": "Sem demonstracoes; usar a classe padrao do mock.",
                "intencao": "informacao",
                "sentimento": "neutro",
            }

        # A chave vazia sempre casa com a mensagem final. O DummyLM nativo cuida
        # do formato esperado pelo adaptador do DSPy.
        self.answers = {"": resposta}
        return super().forward(prompt=prompt, messages=messages, **kwargs)


DATASET = [
    dspy.Example(
        texto="Quero cancelar meu plano hoje. Estou muito insatisfeito.",
        intencao="cancelamento",
        sentimento="negativo",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Cobraram minha assinatura duas vezes. Isso e um absurdo.",
        intencao="cobranca",
        sentimento="negativo",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Nao consigo entrar na conta porque esqueci minha senha.",
        intencao="acesso",
        sentimento="negativo",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Qual e o prazo de entrega do novo cartao?",
        intencao="informacao",
        sentimento="neutro",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Obrigado, o problema foi resolvido rapidamente!",
        intencao="elogio",
        sentimento="positivo",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Preciso alterar o vencimento da minha fatura.",
        intencao="cobranca",
        sentimento="neutro",
    ).with_inputs("texto"),
    # Holdout: estes dois exemplos nao participam da compilacao.
    dspy.Example(
        texto="Minha senha foi bloqueada e estou irritado.",
        intencao="acesso",
        sentimento="negativo",
    ).with_inputs("texto"),
    dspy.Example(
        texto="Excelente atendimento, muito obrigado!",
        intencao="elogio",
        sentimento="positivo",
    ).with_inputs("texto"),
]


def metrica_exata(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Metrica de negocio deterministica: os dois rotulos devem estar corretos."""
    return (
        normalizar(str(prediction.intencao)) == normalizar(str(example.intencao))
        and normalizar(str(prediction.sentimento))
        == normalizar(str(example.sentimento))
    )


def avaliar(programa: dspy.Module, exemplos: list[dspy.Example]) -> tuple[int, list]:
    resultados = []
    acertos = 0
    for example in exemplos:
        prediction = programa(**example.inputs())
        correto = metrica_exata(example, prediction)
        acertos += int(correto)
        resultados.append((example, prediction, correto))
    return acertos, resultados


def quantidade_demos(programa: dspy.Module) -> int:
    """Conta demonstracoes no unico predictor da POC."""
    return len(programa.classificar.predict.demos)


def imprimir_resultados(titulo: str, resultados: list) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))
    for example, prediction, correto in resultados:
        print(f"Texto       : {example.texto}")
        print(f"Esperado    : {example.intencao} / {example.sentimento}")
        print(f"Predito     : {prediction.intencao} / {prediction.sentimento}")
        print(f"Metrica     : {'1 (correto)' if correto else '0 (incorreto)'}")
        print()


def main() -> None:
    random.seed(42)

    mock_lm = DemoAwareDummyLM()
    # Forma solicitada: configuracao global do LM local, sem qualquer API.
    dspy.settings.configure(lm=mock_lm)

    trainset = DATASET[:6]
    holdout = DATASET[6:]

    programa_base = ClassificadorSuporte()
    acertos_base, resultados_base = avaliar(programa_base, holdout)

    optimizer = dspy.BootstrapFewShot(
        metric=metrica_exata,
        max_bootstrapped_demos=2,
        max_labeled_demos=4,
        max_rounds=1,
        max_errors=5,
    )
    programa_compilado = optimizer.compile(
        student=ClassificadorSuporte(),
        trainset=trainset,
    )

    acertos_compilado, resultados_compilado = avaliar(programa_compilado, holdout)

    print("=" * 72)
    print("POC DSPy LOCAL - INTENCAO + SENTIMENTO")
    print("=" * 72)
    print(f"DSPy              : {dspy.__version__}")
    print("LM                 : DemoAwareDummyLM (zero rede / zero API)")
    print(f"Treino / holdout   : {len(trainset)} / {len(holdout)}")
    print(f"Demos antes/depois : {quantidade_demos(programa_base)} / "
          f"{quantidade_demos(programa_compilado)}")

    imprimir_resultados("PIPELINE NAO COMPILADO", resultados_base)
    imprimir_resultados("PIPELINE COMPILADO COM BootstrapFewShot", resultados_compilado)

    print("RESUMO")
    print("------")
    print(f"Nao compilado : {acertos_base}/{len(holdout)}")
    print(f"Compilado     : {acertos_compilado}/{len(holdout)}")

    # Autoteste: se uma mudanca de API quebrar a POC, o processo termina com erro.
    assert quantidade_demos(programa_base) == 0
    assert quantidade_demos(programa_compilado) > 0
    assert acertos_base == 0
    assert acertos_compilado == len(holdout)
    print("\nAUTOTESTE: OK - compilacao, inferencia e metrica validadas.")


if __name__ == "__main__":
    main()
