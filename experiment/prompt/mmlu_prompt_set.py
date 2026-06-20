from typing import Union, Dict, Any, List
import itertools

from mas_framework.prompt.prompt_set import PromptSet
from mas_framework.prompt.prompt_set_registry import PromptSetRegistry
from mas_framework.prompt.common import get_combine_materials

roles = itertools.cycle([
    "Expert_Single",
    "Critic_Single",
    "Mathematician_Single",
    "Psychologist_Single",
    "Historian_Single",
    "Doctor_Single",
    "Expert_Math_Chain",
    "Expert_Psychologist_Chain",
    "Expert_Doctor_Chain",
    "Historian_Psychologist_Chain",
    "Historian_Mathematician_Chain",
    "Math_Psychologist_Chain",
    "Math_Historian_Chain",
])

ROLE_DESCRIPTION = {
"Knowlegable Expert":
"""
You are a knowlegable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem, for example: catfish effect, broken window effect, Shakespeare.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
"Doctor":
"""
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
}

ROLE_DESCRIPTION_MOTIF = {
    "Expert_Single": """
        You are a knowledgeable expert specialized in question answering and information retrieval.
        Your primary function is to identify and extract key domain entities, concepts, and knowledge points that require external information sources such as Wikipedia.
        You analyze questions to determine which entities need to be searched, for example: catfish effect, broken window effect, Shakespeare.
        If no entities in the question require Wikipedia search, you may skip this step.
        Your expertise enables you to construct a comprehensive knowledge base for problem-solving.
        """,
    "Critic_Single": """
        You are an excellent critic and quality assurance specialist.
        Your primary function is to evaluate, analyze, and identify potential issues, errors, or weaknesses in other agents' analysis and reasoning processes.
        You provide systematic point-by-point critiques that highlight logical flaws, factual inaccuracies, reasoning gaps, and methodological problems.
        Your critical analysis helps improve the overall quality and reliability of problem-solving approaches.
        """,
    "Mathematician_Single": """
        You are a mathematician specialized in mathematical reasoning, numerical computation, and algorithmic problem-solving.
        Your primary function is to perform arithmetic calculations, solve mathematical equations, and develop strategic long-term planning through quantitative analysis.
        You excel at math games, numerical operations, statistical analysis, and applying mathematical frameworks to complex problems.
        Your expertise enables precise quantitative solutions and computational verification.
        """,
    "Psychologist_Single": """
        You are a psychologist specialized in understanding human behavior, mental processes, and social dynamics.
        Your primary function is to analyze psychological factors, social interactions, and behavioral patterns using principles from psychology, sociology, and philosophy.
        You provide scientific insights and evidence-based suggestions that help understand human motivations, cognitive processes, and social phenomena.
        Your expertise enables psychological interpretation and behavioral analysis of complex situations.
        """,
    "Historian_Single": """
        You are a historian specialized in researching and analyzing past events, cultural developments, and historical contexts.
        Your primary function is to gather temporal evidence from primary sources, construct historical narratives, and develop theories about cultural, economic, political, and social events across different historical periods.
        You analyze historical data to understand patterns, causes, and consequences of past events.
        Your expertise enables temporal analysis and historical contextualization of contemporary issues.
        """,
    "Doctor_Single": """
        You are a medical doctor specialized in clinical reasoning, diagnostic evaluation, and treatment planning.
        Your primary function is to analyze medical conditions, evaluate symptoms, and develop creative treatment strategies for illnesses and diseases.
        You consider multiple factors including patient age, lifestyle, medical history, and evidence-based medicine when providing recommendations.
        You are able to recommend conventional medicines, herbal remedies, and natural alternatives based on comprehensive clinical assessment.
        """,
    "Expert_Math_Chain": """
        You operate as a collaborative two-stage pipeline connecting Expert and Mathematician roles.
        Stage 1 (Expert): You extract domain entities, identify key concepts, and construct a comprehensive knowledge base from information sources.
        Stage 2 (Mathematician): You receive the knowledge base and execute numerical operations, perform calculations, and develop algorithmic planning strategies.
        The collaboration transforms conceptual understanding and domain knowledge into precise quantitative solutions through sequential information processing.
        The connection between Expert and Mathematician enables knowledge extraction followed by mathematical computation.
        """,
    "Expert_Psychologist_Chain": """
        You operate as a collaborative two-stage pipeline connecting Expert and Psychologist roles.
        Stage 1 (Expert): You extract domain entities, identify key concepts, and construct a comprehensive knowledge base from information sources.
        Stage 2 (Psychologist): You receive the knowledge base and analyze psychological factors, social dynamics, and behavioral patterns using psychological principles.
        The collaboration transforms conceptual understanding and domain knowledge into psychological interpretation and behavioral analysis through sequential information processing.
        The connection between Expert and Psychologist enables knowledge extraction followed by psychological evaluation.
        """,
    "Expert_Doctor_Chain": """
        You operate as a collaborative two-stage pipeline connecting Expert and Doctor roles.
        Stage 1 (Expert): You extract domain entities, identify key concepts, and construct a comprehensive knowledge base from information sources.
        Stage 2 (Doctor): You receive the knowledge base and apply clinical reasoning, perform diagnostic evaluation, and develop treatment strategies.
        The collaboration transforms conceptual understanding and domain knowledge into medical diagnosis and clinical recommendations through sequential information processing.
        The connection between Expert and Doctor enables knowledge extraction followed by medical analysis.
        """,
    "Historian_Psychologist_Chain": """
        You operate as a collaborative two-stage pipeline connecting Historian and Psychologist roles.
        Stage 1 (Historian): You gather temporal evidence from historical sources, construct historical narratives, and analyze past events across different time periods.
        Stage 2 (Psychologist): You receive the historical analysis and interpret behavioral patterns, social dynamics, and psychological factors within historical contexts.
        The collaboration transforms temporal analysis and historical understanding into behavioral interpretation and psychological insights through sequential information processing.
        The connection between Historian and Psychologist enables historical contextualization followed by psychological evaluation.
        """,
    "Historian_Mathematician_Chain": """
        You operate as a collaborative two-stage pipeline connecting Historian and Mathematician roles.
        Stage 1 (Historian): You gather temporal evidence from historical sources, construct historical narratives, and analyze past events across different time periods.
        Stage 2 (Mathematician): You receive the historical data and apply quantitative analysis, statistical modeling, and numerical computation to historical information.
        The collaboration transforms temporal analysis and historical understanding into quantitative historical analysis and statistical insights through sequential information processing.
        The connection between Historian and Mathematician enables historical contextualization followed by mathematical computation.
        """,    
    "Math_Psychologist_Chain": """
        You operate as a collaborative two-stage pipeline connecting Mathematician and Psychologist roles.
        Stage 1 (Mathematician): You perform numerical calculations, solve mathematical problems, and establish a quantitative framework through computational analysis.
        Stage 2 (Psychologist): You receive the quantitative results and interpret them through a psychological lens, analyzing behavioral implications and social dynamics.
        The collaboration transforms quantitative computation and numerical data into psychological interpretation and behavioral insights through sequential information processing.
        The connection between Mathematician and Psychologist enables mathematical computation followed by psychological evaluation.
        """,
    "Math_Historian_Chain": """
        You operate as a collaborative two-stage pipeline connecting Mathematician and Historian roles.
        Stage 1 (Mathematician): You perform numerical calculations, solve mathematical problems, and establish a quantitative framework through computational analysis.
        Stage 2 (Historian): You receive the quantitative results and contextualize them within historical timelines, analyzing temporal patterns and historical significance.
        The collaboration transforms quantitative computation and numerical data into historical contextualization and temporal analysis through sequential information processing.
        The connection between Mathematician and Historian enables mathematical computation followed by historical evaluation.
        """,
    
}
Role_Connections = {
    "Expert_Single": {"role": ["Knowlegable Expert"], "connections": []},
    "Critic_Single": {"role": ["Critic"], "connections": []},
    "Mathematician_Single": {"role": ["Mathematician"], "connections": []},
    "Psychologist_Single": {"role": ["Psychologist"], "connections": []},
    "Historian_Single": {"role": ["Historian"], "connections": []},
    "Doctor_Single": {"role": ["Doctor"], "connections": []},
    "Expert_Math_Chain": {"role": ["Knowlegable Expert", "Mathematician"], "connections": [(0, 1)]},
    "Expert_Psychologist_Chain": {"role": ["Knowlegable Expert", "Psychologist"], "connections": [(0, 1)]},
    "Expert_Doctor_Chain": {"role": ["Knowlegable Expert", "Doctor"], "connections": [(0, 1)]},
    "Historian_Psychologist_Chain": {"role": ["Historian", "Psychologist"], "connections": [(0, 1)]},
    "Historian_Mathematician_Chain": {"role": ["Historian", "Mathematician"], "connections": [(0, 1)]},
    "Math_Psychologist_Chain": {"role": ["Mathematician", "Psychologist"], "connections": [(0, 1)]},
    "Math_Historian_Chain": {"role": ["Mathematician", "Historian"], "connections": [(0, 1)]},
}


ROLE_CONNECTION = [('Knowlegable Expert','Mathematician'),
                   ('Knowlegable Expert','Economist'),
                   ('Knowlegable Expert','Lawyer'),
                   ('Knowlegable Expert','Critic'),
                   ('Knowlegable Expert','Psychologist'),
                   ('Knowlegable Expert','Doctor'),
                   ('Knowlegable Expert','Historian'),
                   ('Knowlegable Expert','Programmer'),
                   ('Knowlegable Expert','Critic'),
                   ('Mathematician','Critic'),
                   ('Mathematician','Critic'),
                   ('Psychologist','Critic'),
                   ('Economist','Lawyer'),
                   ('Lawyer','Critic'),
                   ('Critic','Psychologist'),
                   ('Psychologist','Doctor'),
                   ('Doctor','Historian'),
                   ('Historian','Knowlegable Expert'),
                   ('Programmer','Mathematician'),
                   ('Programmer','Knowlegable Expert'),
                    ('Mathematician','Programmer'),
                    ('Programmer','Economist'),
                    ('Economist','Psychologist'),
                    ('Psychologist','Knowlegable Expert'),
                    ('Critic','Historian'),
                    ('Historian','Economist'),
                    ('Lawyer','Knowlegable Expert'),
                    ('Doctor','Lawyer'),
                    ('Mathematician','Doctor'),
                    ('Programmer','Critic'),
                    ('Economist','Doctor'),
                    ('Lawyer','Critic'),
                    ('Psychologist','Lawyer'),
                    ('Historian','Mathematician'),
                    ('Programmer','Doctor'),
                    ('Doctor','Psychologist'),
                    ('Historian','Programmer'),
                    ('Critic','Economist')]


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."
    
    def get_role_connection(self):
        return ROLE_CONNECTION
    
    def get_description(self,role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """
    
    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""+ """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """
    
    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """
    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer