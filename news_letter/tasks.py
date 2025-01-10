from datetime import datetime
from crewai import Task

class AINewsLetterTasks():
    def fetch_news_task(self, agent):
        return Task(
            description=f'Buscar as principais notícias de IA das últimas 24 horas. O horário atual é {datetime.now()}.',
            agent=agent,
            async_execution=False,
            expected_output="""Uma lista dos principais títulos de notícias de IA, URLs e um breve resumo de cada história das últimas 24 horas.
                Exemplo de Saída:
                [
                    {  'título': 'IA é destaque nos comerciais do Super Bowl',
                    'url': 'https://example.com/story1',
                    'resumo': 'A IA fez sucesso nos comerciais do Super Bowl deste ano...'
                    },
                    {{...}}
                ]
            """
        )

    def analyze_news_task(self, agent, context):
        return Task(
            description='Analisar cada notícia e garantir que haja pelo menos 5 artigos bem formatados',
            agent=agent,
            async_execution=False,
            context=context,
            expected_output="""Uma análise formatada em markdown para cada notícia, incluindo um resumo, pontos detalhados e uma seção "Por que isso é importante". Deve haver pelo menos 5 artigos, cada um seguindo o formato adequado.
                Exemplo de Saída:
                '## IA é destaque nos comerciais do Super Bowl\n\n
                **O Resumo:** A IA fez sucesso nos comerciais do Super Bowl deste ano...\n\n
                **Os detalhes:**\n\n
                - O comercial do Copilot da Microsoft apresentou seu assistente de IA...\n\n
                **Por que isso é importante:** Embora os anúncios relacionados à IA tenham sido abundantes no último ano, sua presença no Super Bowl é um grande momento mainstream.\n\n'
            """
        )

    def compile_newsletter_task(self, agent, context, callback_function):
        return Task(
            description='Compilar o boletim informativo',
            agent=agent,
            context=context,
            expected_output="""Um boletim informativo completo em formato markdown, com um estilo e layout consistentes.
                Exemplo de Saída:
                '# Principais notícias de IA hoje:\\n\\n
                - IA é destaque nos comerciais do Super Bowl\\n
                - Altman busca TRILHÕES para iniciativa global de chips de IA\\n\\n

                ## IA é destaque nos comerciais do Super Bowl\\n\\n
                **O Resumo:** A IA fez sucesso nos comerciais do Super Bowl deste ano...\\n\\n
                **Os detalhes:**...\\n\\n
                **Por que isso é importante::**...\\n\\n
                ## Altman busca TRILHÕES para iniciativa global de chips de IA\\n\\n
                **O Resumo:** O CEO da OpenAI, Sam Altman, está supostamente tentando arrecadar TRILHÕES de dólares...\\n\\n'
                **Os detalhes:**...\\n\\n
                **Por que isso é importante::**...\\n\\n
            """,
            callback=callback_function
        )
