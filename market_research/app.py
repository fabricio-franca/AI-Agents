from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

model = ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini")


search_tool = SerperDevTool(api_key=serper_api_key)


fundamental_analyst = Agent(   
    role="Analista de investimentos de Criptomoedas Fundamentalista",
    goal="Analisar as projeções da moeda Etherium",
    backstory="""Você é um analista de investimentos em criptomoedas sênior especializado em análise fundamentalista.
    Seu trabalho é avaliar as projeções de mercado dessa criptomoeda e seu valor intrínseco baseado no que está sendo desenvolvido
    dentro daquela blockchain, usabilidade da rede ou tudo que esteja ligado àquela rede blockchain etc.""",
    verbose=True,
    allow_delegation=False,
    llm=model,
    tools=[search_tool]
)

tecnical_analyst = Agent(
    role = "Analista Técnico de Criptomoedas",
    goal = "Analisar os padrões gráficos e indicadores técnicos da Etherium",
    backstory = """Você é analista sênior especializado em análise de gráficos de criptomoedas.
    Seu trabalho é indicar padrões de preço e volume, além de usar indicadores técnicos para prever movimentos futuros da cripto.
    """,
    verbose=True,
    allow_delegation=False,
    llm = model,
    tools=[search_tool]
)

tarefa_fundamentalista = Task(
    description="""
    realizar uma análise aprofundada da criptomoeda Ethereum (ETH), avaliando seus fundamentos e valor intrínseco. Utilize dados atualizados e confiáveis para fornecer uma visão clara do potencial do Ethereum como ativo de investimento.

    Para sua análise, siga os seguintes pontos:

    1. Visão Geral do Projeto:
    História: Analise o histórico do Ethereum desde seu lançamento em 2015, destacando os marcos importantes no desenvolvimento do projeto.
    Equipe: Analise informações sobre os fundadores, incluindo Vitalik Buterin, e outros membros-chave do time.
    Problema Resolvido: Analise como o Ethereum introduziu contratos inteligentes e o conceito de "computador mundial".
    Inovações e Atualizações: Avalie atualizações recentes (ex.: Ethereum Merge, transição para Proof of Stake) e futuras melhorias planejadas (ex.: sharding).
    2. Tokenomics:
    Oferta e Emissão: Analise a atual oferta circulante de ETH, modelo de emissão pós-Merge (com queima de ETH), e seu impacto na inflação/deflação do token.
    Casos de Uso: Analise como o ETH é usado como combustível para contratos inteligentes, staking, DeFi, NFTs e outros aplicativos no ecossistema Ethereum.
    Distribuição: Avalie a concentração de ETH entre grandes carteiras (baleias) e investidores institucionais.
    3. Parcerias e Adoção:
    Parcerias: Analise colaborações importantes (ex.: empresas que utilizam a Ethereum Virtual Machine).
    Integrações: Identifique empresas ou governos que utilizam Ethereum para projetos de blockchain.
    Adoção: Analise métricas como o número de dApps no Ethereum, TVL (Total Value Locked) em DeFi, e volumes de transação em NFTs.
    4. Comunidade e Desenvolvimento:
    Engajamento: Avalie o tamanho e a atividade da comunidade no Twitter, Reddit, Discord e GitHub.
    Desenvolvedores: Verifique o número de contribuidores ativos no repositório do Ethereum no GitHub.
    Projetos no Ecossistema: Analise o crescimento de novas aplicações, como Layer-2 (Polygon, Optimism, Arbitrum).
    5. Riscos e Desafios:
    Regulamentação: Analise os impactos de potenciais regulamentações globais sobre o Ethereum.
    Concorrência: Avalie a concorrência com blockchains como Solana, Binance Smart Chain e Cardano.
    Segurança: Examine incidentes de segurança no passado, como hacks em contratos inteligentes ou exploits em dApps.
    """,
    expected_output="Relatório completo de avaliação da moeda, com previsões de movimentações para os próximos 3 meses.",
    agent=fundamental_analyst
)

tarefa_tecnica = Task(
    description="""
    Você é um analista técnico especializado em criptomoedas. Seu objetivo é realizar uma análise técnica detalhada do preço do Ethereum (ETH), utilizando ferramentas de análise gráfica e indicadores técnicos para identificar possíveis movimentos futuros do mercado. Sua análise deve considerar múltiplos períodos de tempo e ser baseada em dados confiáveis.

    1. Contextualização do Mercado:
    Análise Geral:
    Avalie a tendência geral do Ethereum no curto, médio e longo prazo (alta, baixa ou lateralidade).
    Condições de Mercado:
    Observe o contexto geral do mercado de criptomoedas, incluindo volatilidade e correlação com o Bitcoin (BTC).
    2. Análise de Gráficos:
    Identifique Tendências Primárias:
    Use gráficos de velas para períodos diários, semanais e mensais.
    Determine máximas e mínimas recentes.
    Trace linhas de tendência (alta ou baixa).
    Zonas de Suporte e Resistência:
    Identifique níveis críticos de suporte e resistência com base em preços históricos significativos.
    Use volumes para validar a força desses níveis.
    3. Indicadores Técnicos:
    Indicadores de Momentum:
    RSI (Índice de Força Relativa): Identifique sobrecompra ou sobrevenda (níveis acima de 70 ou abaixo de 30).
    MACD (Média Móvel de Convergência/Divergência): Analise cruzamentos para identificar reversões de tendência.
    Médias Móveis:
    Acompanhe médias móveis simples e exponenciais (SMA e EMA) de 20, 50 e 200 períodos para identificar tendências predominantes.
    Indicadores de Volume:
    Observe o OBV (On-Balance Volume) para confirmar movimentos de preço.
    Utilize o indicador de volume profile para identificar zonas de alto interesse.
    4. Padrões Gráficos:
    Procure por padrões técnicos como:
    Triângulos (simétricos, ascendentes ou descendentes).
    Ombro-Cabeça-Ombro (OCO).
    Bandeiras ou flâmulas indicando continuação de tendência.
    5. Fibonacci e Níveis de Expansão:
    Utilize retrações e extensões de Fibonacci para identificar potenciais áreas de reversão ou continuação de tendência.
    Aplique Fibonacci nos movimentos de alta ou baixa recentes para projetar possíveis alvos de preço.
    6. Análise de Velas (Price Action):
    Observe formações de velas individuais para identificar sinais de reversão ou continuação (ex.: martelo, estrela cadente, engolfo de alta/baixa).
    Combine a leitura das velas com os níveis de suporte e resistência.
    7. Métricas On-Chain: (opcional, se necessário)
    Compare dados técnicos com métricas on-chain, como:
    Endereços ativos.
    Taxas de transação média.
    Crescimento do supply em staking.
    """,
    expected_output="Relatório completo de avaliação da moeda, com previsões de movimentações para os próximos 3 meses",
    agent=tecnical_analyst
)

crew = Crew(
    agents=[fundamental_analyst,tecnical_analyst],
    tasks=[tarefa_fundamentalista,tarefa_tecnica],
    verbose=True,
    process=Process.sequential
)

result = crew.kickoff(inputs={'input':'Etherium'})

print("-----------------------")

print(result)