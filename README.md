# POC DSPy 100% local: classificação de suporte

Esta POC demonstra, de forma didática, os quatro pilares do DSPy e a diferença entre um programa não compilado e outro compilado com `BootstrapFewShot`. Depois de instalar o pacote, a execução não usa internet, chave de API, endpoint pago, servidor local nem download de modelo.

Compatibilidade validada: Python 3.10+ e `dspy-ai==3.3.0` (o metapacote instala `dspy==3.3.0`).

## 1. Visão executiva

DSPy é um framework para construir sistemas de IA como programas estruturados. Em vez de manter uma grande string de prompt e editá-la manualmente, o desenvolvedor declara:

- quais dados entram e saem;
- como as etapas são compostas;
- como a qualidade será medida;
- qual otimizador escolherá instruções ou exemplos para o programa.

### Analogia com um compilador

Em uma linguagem tradicional, o programador descreve a lógica e o compilador transforma essa descrição em uma representação adequada à máquina. No DSPy, o desenvolvedor descreve o contrato e a arquitetura do pipeline; o otimizador transforma dados rotulados e métricas em uma configuração de prompts e demonstrações mais adequada ao LM.

Outra analogia é um otimizador de rotas: você informa os destinos, restrições e a métrica — por exemplo, menor custo com SLA cumprido. O sistema procura uma rota. No DSPy, os destinos são os outputs corretos, as restrições estão nas signatures e a função objetivo é a métrica.

DSPy não “compila Python para código de máquina”. A compilação de um programa DSPy significa otimizar parâmetros do pipeline, como instruções e exemplos few-shot, orientada por dados e métricas.

### Benefícios tangíveis

- **Reprodutibilidade:** dataset, métrica, versão do LM, configuração do otimizador e artefato compilado podem ser versionados.
- **Manutenção desacoplada:** o contrato e a composição do programa ficam separados do texto final enviado ao LM.
- **Otimização por objetivo de negócio:** a seleção deixa de depender apenas de “este prompt parece melhor” e passa a maximizar precisão, conformidade, recall, custo ou outra métrica explícita.
- **Troca controlada de modelo:** é possível recompilar e reavaliar o mesmo programa quando o LM muda.

## 2. Arquitetura e conceitos-chave

### Os quatro pilares

1. **Signatures:** contratos de entrada e saída. Nesta POC, `ClassificarSuporte` recebe `texto` e devolve `intencao` e `sentimento`, ambos com domínios tipados por `Literal`.
2. **Modules:** componentes executáveis e combináveis. `ClassificadorSuporte` deriva de `dspy.Module` e encapsula `dspy.ChainOfThought`.
3. **Metrics:** funções que comparam resultado esperado e predição. `metrica_exata` retorna verdadeiro somente quando os dois rótulos coincidem.
4. **Optimizers/Teleprompters:** algoritmos que compilam parâmetros do programa. `BootstrapFewShot` coleta demonstrações, filtra traces pela métrica e adiciona exemplos rotulados.

### Engenharia manual versus DSPy

| Dimensão | Engenharia de prompts manual | Programação declarativa com DSPy |
|---|---|---|
| Unidade principal | String de prompt | Signature + Module |
| Contrato de I/O | Implícito no texto | Campos declarados e tipados |
| Evolução | Edição manual e tentativa/erro | Compilação com dataset e métrica |
| Avaliação | Frequentemente subjetiva | Função reexecutável e versionável |
| Few-shot | Exemplos copiados à mão | Demos selecionadas pelo otimizador |
| Troca de LM | Pode exigir reescrever prompts | Recompilar e comparar métricas |
| Manutenção | Texto, lógica e formato acoplados | Estrutura separada da otimização |
| Governança | Difícil explicar por que mudou | Configuração, dados e score auditáveis |

## 3. O que o código demonstra

O dataset tem oito `dspy.Example`: seis para compilação e dois para holdout. O `DemoAwareDummyLM` deriva do `DummyLM` nativo e é determinístico:

- sem demonstrações compiladas, responde com uma classe padrão fraca;
- quando o DSPy injeta demonstrações, aplica regras locais simples;
- nunca chama rede, API, Hugging Face, Ollama ou outro processo.

Essa construção permite provar localmente o fluxo de `Signature -> Module -> Metric -> Optimizer`. Ela **não** mede a inteligência nem a capacidade de generalização de um LM real; o ganho impresso é uma demonstração mecânica e controlada da compilação.

O `BootstrapFewShot` é real, não uma função falsa com o mesmo nome. A simulação está apenas no LM, exatamente para retirar custo, autenticação, latência e variabilidade.

## 4. Execução local

### Terminal

Na pasta que contém os arquivos:

```bash
python -m pip install "dspy-ai==3.3.0"
python dspy_poc_local.py
```

O primeiro comando é a única etapa que precisa de internet para obter o pacote. A execução do segundo comando é inteiramente offline.

### Jupyter Notebook

Em uma célula:

```python
%pip install "dspy-ai==3.3.0"
```

Reinicie o kernel se o ambiente solicitar. Em seguida, execute:

```python
%run dspy_poc_local.py
```

Não defina `OPENAI_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN` ou qualquer outra credencial. O script configura o mock com `dspy.settings.configure(lm=mock_lm)`.

## 5. Resultado esperado

O console deve mostrar:

- zero demonstrações antes da compilação e uma ou mais depois;
- pipeline não compilado com `0/2` no holdout;
- pipeline compilado com `2/2` no holdout;
- `AUTOTESTE: OK` ao final.

Durante a compilação, o DSPy também pode exibir uma barra de progresso curta. Não há timeout de rede, pois nenhuma inferência sai do processo Python.

## 6. Limites e caminho para produção

- O mock valida a arquitetura e os contratos, não a qualidade de um modelo real.
- Com um LM real, o holdout deve ser maior, estratificado e totalmente separado do treino.
- Métricas produtivas podem combinar acurácia por classe, recall de intenções críticas, custo, latência e regras de compliance.
- Antes de promover um programa compilado, fixe versões, salve seu estado, execute regressão e mantenha um baseline não compilado.

## Referências oficiais

- DSPy: https://dspy.ai/
- Instalação: https://dspy.ai/getting-started/installation/
- BootstrapFewShot: https://dspy.ai/api/optimizers/BootstrapFewShot/
- Família BootstrapFewShot: https://dspy.ai/diving-deeper/bootstrap-fewshot-family/
