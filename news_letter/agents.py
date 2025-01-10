from crewai import Agent
from tools.search_tools import SearchTools

class AINewsLetterAgents():
    def editor_agent(self):
        return Agent(
            role='Editor',
            goal='Supervisionar a criação do Boletim Informativo de IA',
            backstory="""Com um olhar atento aos detalhes e uma paixão por contar histórias, você garante que o boletim
            não apenas informe, mas também envolva e inspire os leitores.""",
            allow_delegation=True,
            verbose=True,
            max_iter=15
        )

    def news_fetcher_agent(self):
        return Agent(
            role='ColetorDeNotícias',
            goal='Buscar as principais notícias de IA do dia',
            backstory="""Como um detetive digital, você vasculha a internet em busca dos desenvolvimentos mais recentes
            e impactantes no mundo da IA, garantindo que nossos leitores estejam sempre bem informados.""",
            tools=[SearchTools.search_internet],  # Usando a função do SearchTools
            verbose=True,
            allow_delegation=True,
        )

    def news_analyzer_agent(self):
        return Agent(
            role='AnalisadorDeNotícias',
            goal='Analisar cada notícia e gerar um resumo detalhado em md',
            backstory="""Com um olhar crítico e uma habilidade para destilar informações complexas, você fornece análises
            perspicazes das notícias de IA, tornando-as acessíveis e envolventes para o nosso público.""",
            tools=[SearchTools.search_internet],  # Usando a função do SearchTools
            verbose=True,
            allow_delegation=True,
        )

    def newsletter_compiler_agent(self):
        return Agent(
            role='CompiladorDeBoletim',
            goal='Compilar as notícias analisadas em um formato final de boletim',
            backstory="""Como o arquiteto final do boletim, você organiza e formata meticulosamente o conteúdo,
            garantindo uma apresentação coerente e visualmente atraente que cativa nossos leitores. Certifique-se de seguir
            as diretrizes de formato do boletim e manter a consistência ao longo do texto.""",
            verbose=True,
        )
