import re
import os
import sys
import time
from typing import Dict, TypedDict

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from httpx import Timeout

from src.tools.startup import logger
from src.data.prompts import (generation_prompt,
                              context_prompt,
                              generation_prompt_app,
                              context_prompt_app)
from src.data.get_chroma_db import get_chroma_db
from googletrans import Translator


IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
AWS_DEPLOYMENT = bool(os.environ.get("AWS_DEPLOYMENT", False))


class RAGRefugeesChatbot:
    """
    Class that encapsulates refugees' chatbot functionality.
    This is powered by Retrieval-Augmented-Generation (RAG) techniques.
    """
    def __init__(self, settings: dict) -> None:
        """
        Initizalize class. This does the following:
        - Set internal attributes
        - Initialize the retriever
        - Initialize the generator
        - Define the RAG graph

        Args:
            settings (dict): Class settings.
        """
        self._default_answer = settings['default_answer']
        if not AWS_DEPLOYMENT:
            import torch
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            else:
                self._device = torch.device('cpu')
        self.translator = self._initialize_language_detector_and_translator()
        self._context_response = settings['context_response']
        self.retriever = self._initialize_retriever(**settings['retrieval'])
        self.llm_mode = settings["llm_mode"]
        tokenizer = None
        if self.llm_mode == "api" or AWS_DEPLOYMENT:
            self.model = ChatBedrock(model_id=settings["api"]["bedrock_model"])
            self.prompt = {
                'generation': ChatPromptTemplate(generation_prompt_app),
                'context': ChatPromptTemplate(context_prompt_app),
            }
        elif self.llm_mode == "local":
            self.model, tokenizer = self._initialize_llm(**settings['llm'])
            self.prompt = {
                'generation': self._initialize_prompt(generation_prompt, tokenizer),
                'context': self._initialize_prompt(context_prompt, tokenizer)
            }
        else:
            raise NameError(f"llm_mode must be one of \"local\" or \"api\" but "
                            f"\"{self.llm_mode}\" was received")
        
        self.graph = self._define_graph()

    def _initialize_language_detector_and_translator(self) -> object:
        """
        Initialize language detection module

        Returns:
            Language detector and translator object.
        """
        translator = Translator(timeout=Timeout(20))
        return translator

    def _initialize_prompt(
            self, prompt_content: str, tokenizer: object) -> object:
        """
        Initialize prompt. This implies reading the prompt and using the
        model's tokenizer to apply the prompt template of each specific model.

        Args:
            prompt_content (str): Content of the prompt, independent of the template.
            tokenizer (object): model's tokenizer object.

        Returns:
            Prompt object
        """
        prompt_template = tokenizer.apply_chat_template(
            prompt_content, tokenize=False, add_generation_prompt=True)
        prompt = PromptTemplate.from_template(template=prompt_template)
            
        return prompt

    def _initialize_retriever(
            self, model_name: str, persist_directory: str, collection_name: str
            ) -> object:
        """
        Initialize document retriever. This implies the following steps:
        - Set embedding model
        - Load the indexed vectorstore
        - Initialize the retriever as a Chroma DB

        Args:
            model_name (str): Name of model to generate embeddings from documents.
            persist_directory (str): Directory where the vector database is stored.
            collection_name (str): Name of retriever collection.

        Returns:
            Document retriever object.

        Raises:
            NameError: when the specified vectorstore does not exist
        """
        # Setup model
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        if IS_USING_IMAGE_RUNTIME:
            # Get runtime and if running from Lambda, copy the database
            # to temporal directory to have write permissions
            persist_directory = get_chroma_db(persist_directory)

        else:
            # Check if the vectorstore exists, raise an error if it does not
            vectorstore_configs_file = \
                f'{persist_directory}_{collection_name}_configs.json'
            if not os.path.exists(vectorstore_configs_file):
                raise NameError(
                    f"The specified vectorstore \"{persist_directory}\" and "\
                    f"collection \"{collection_name}\" do not exist, please create "\
                    f"it first.")

        # Load vector store
        if IS_USING_IMAGE_RUNTIME:
            __import__("pysqlite3")
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

        from langchain_chroma import Chroma
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        retriever = vectorstore.as_retriever()
        return retriever

    def _initialize_llm(self, model_name: str, revision: str, pipeline: dict,
                        quantization_configs: dict) -> object:
        """
        Initialize text generator. This implies the following steps:
        - Define autotokenizer and model for text generation
        - Instantiate LLM pipeline

        Args:
            model_name (str): Name of model to generate text.
            revision (str): Name of the branch of the model to use.
            pipeline (dict): Parameters used to define the LLM pipeline.
            quantization_configs (dict): Quantization configurations.

        Returns:
            Text generator object.
        """
        import torch
        import transformers
        from transformers import (AutoModelForCausalLM, 
                                  AutoTokenizer, 
                                  BitsAndBytesConfig)

        # Prepare model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model_configs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": "auto",
            "trust_remote_code": False,
            "torch_dtype": torch.float16,
            "revision": revision
        }

        if isinstance(quantization_configs, dict):
            model_configs["quantization_config"] = \
                BitsAndBytesConfig(**quantization_configs)

        model = AutoModelForCausalLM.from_pretrained(**model_configs)

        # Define pipeline
        pipeline_configs = {
            "model": model,
            "tokenizer": tokenizer,
            "task": "text-generation",
            **pipeline}

        if quantization_configs is None:
            model.to(self._device)

        text_generation_pipeline = transformers.pipeline(**pipeline_configs)
        llm_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm_pipeline, tokenizer

    def _define_graph(self) -> object:
        """
        Define RAG graph.

        Returns:
            Graph representing RAG flow.
        """

        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                keys: A dictionary where each key is a string.
            """
            keys: Dict[str, any]

        # Define nodes
        workflow = StateGraph(GraphState)
        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("translate_question", self.translate_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("answer_no_context", self.return_no_context_answer)
        workflow.add_node("gather_answer", self.pass_action)
        workflow.add_node("translate_answer", self.translate_answer)
        # Define flow
        workflow.set_entry_point("detect_language")
        workflow.add_conditional_edges(
            "detect_language",
            self.decide_to_translate,
            {
                True: "translate_question",
                False: "retrieve",
            },
        )
        workflow.add_edge("translate_question", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.detect_context,
            {
                True: "generate",
                False: "answer_no_context",
            },
        )
        workflow.add_edge("generate", "gather_answer")
        workflow.add_edge("answer_no_context", "gather_answer")
        workflow.add_conditional_edges(
            "gather_answer",
            self.decide_to_translate,
            {
                True: "translate_answer",
                False: END,
            },
        )
        workflow.add_edge("translate_answer", END)
        chatbot = workflow.compile()
        return chatbot

    def detect_language(self, state: dict) -> dict:
        """
        Identify question language

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, question_language,
                that contains identified question language.
        """
        logger.debug("Identifying question language")
        state_dict = state["keys"]
        question = state_dict["question"]
        language = self.translator.detect(question).lang
        state = {"keys": {"question": question, "language": language}}
        return state

    def translate_question(self, state: dict) -> dict:
        """
        Translate the question from the original language to Spanish

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Question key modified in state: translated to Spanish
        """
        logger.debug('Translating question')
        state_dict = state["keys"]
        question = state_dict["question"]
        language = state_dict["language"]
        translated_question = self.translator.translate(
            question, src=language, dest='es').text
        state = {
            "keys": {
                "question": translated_question,
                "language": language
            }
        }
        return state

    @staticmethod
    def format_docs(docs: list) -> str:
        """
        Format documents before adding them to the prompt

        Args:
            docs (list): List of docs without formatting.

        Returns:
            (str): Formatted documents.
        """
        return "\n\n".join(f"{doc.page_content}"
                           for i, doc in enumerate(docs, start=1))

    def detect_context(self, state: dict) -> bool:
        """
        Detect whether the question asked corresponds to the same context as
        the documents used by the RAG, reflected in the reference embeddings.

        Args:
            state (dict): The current graph state

        Returns:
            bool: Whether the question is within the expected context.
        """
        logger.debug('Determining whether the question is within context')
        logger.debug('Context detection - LLM call')
        documents = state['keys']['documents']
        question = state['keys']['question']
        # Define chain
        chain = self.prompt['context'] | self.model | StrOutputParser()
        # Run
        if self.llm_mode == "api" and not IS_USING_IMAGE_RUNTIME:
            time.sleep(4)
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = chain.invoke(
                    {"documents": self.format_docs(documents),
                     "question": question})
                break # exit loop if successful
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(20) # delay between retries

        full_prompt = self.prompt['context'].format(
            documents=self.format_docs(documents), question=question)
        response = response.replace(full_prompt, "")

        logger.debug('Context detection - LLM answer: "%s"', response)
        logger.debug('Context detection - Matching regular expression')
        is_context = self.match_regular_expression(
            response, **self._context_response)
        return is_context

    def detect_context_node(self, state: dict) -> dict:
        is_context = self.detect_context(state)
        state['keys']['is_context'] = is_context
        return state

    @staticmethod
    def match_regular_expression(
            text: str, regexp: str, positive: bool) -> bool:
        """
        Match regular expression and provide the associated boolean.
        The reason for not using a plain matcher, is that it should work for
        two different cases:

        - Search for regular expressions that represent "positive" cases; in
          this case, a match would result in `True`
        - Search for regular expressions that represent "negative" cases; in
          this case, a match would result in `False`

        Args:
            text (str): Input text.
            regexps (str): Regular expression to match
            positive (bool): Whether to consider a match as positive.

        Returns:
            bool: Boolean aligned with the definition provided.

        Note:
            Newlines are removed before applying the regular expression.
        """
        text = text.replace('\n', '')
        is_match = bool(re.match(regexp, text))
        if positive:
            result = is_match
        else:
            result = not is_match
        return result

    def return_no_context_answer(self, state: dict) -> dict:
        """
        Generate answer to question that is outside the context.

        Args:
            state (dict): Current state of the graph.

        Returns:
            dict: Updated state with default answer for question outside context.
        """
        logger.debug('Returning answer for question outside context')
        state = {
            "keys": {
                "question": state['keys']['question'],
                "answer": self._default_answer,
                "language": state['keys']['language']}
        }
        return state

    def pass_action(self, state: dict) -> dict:
        """
        Action to silently pass, without doing anything.

        Args:
            state (dict): Current state of the graph.

        Returns:
            dict: State of the graph as is
        """
        return state

    def retrieve(self, state: dict) -> dict:
        """
        Retrieve information from documents.

        Args:
            state (dict): Current state of the graph.

        Returns:
            dict: Updated state with documents added.
        """
        logger.debug("Retrieving information from documents")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.retriever.invoke(question)
        state = {
            "keys": {
                "documents": documents,
                "question": question,
                "language": state_dict["language"]
            }
        }
        return state

    def generate(self, state: dict) -> dict:
        """
        Generate answer to question. This is done by:
        1. Getting the prompt from a template
        2. Defining pipeline
        3. Invoking the pipeline

        Args:
            state (dict): Current state of the graph.

        Returns:
            dict: Updated state with answer added.
        """
        logger.debug("Generating answer")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        language = state_dict["language"]
        # Define chain
        chain = self.prompt['generation'] | self.model | StrOutputParser()
        # Run
        if self.llm_mode == "api" and not IS_USING_IMAGE_RUNTIME:
            time.sleep(4)
        retries = 3
        generation = None
        for attempt in range(retries):
            try:
                generation = chain.invoke(
                    {"documents": self.format_docs(documents),
                     "question": question})
                break # exit loop if successful
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(20) # delay between retries

        full_prompt = self.prompt['generation'].format(
            documents=self.format_docs(documents), question=question)
        generation = generation.replace(full_prompt, "")

        state = {
            "keys": {
                "documents": documents,
                "question": question,
                "answer": generation,
                "language": language
            }
        }
        return state

    def translate_answer(self, state: dict) -> dict:
        """
        Translate the answer from Spanish to the original language of the
        question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Generation key modified in state: translated to the
                language of the question.
        """
        logger.debug('Translating answer')
        state_dict = state["keys"]
        generation = state_dict["answer"]
        language = state_dict["language"]
        translated_generation = self.translator.translate(
            generation, dest=language).text
        state = {
            "keys": {
                "question": state_dict["question"],
                "answer": translated_generation,
                "language": language
            }
        }
        if "documents" in state_dict:
            state["keys"]["documents"] = state_dict["documents"]
        return state

    def decide_to_translate(self, state: dict) -> bool:
        """
        Conditional Edge. Determine whether to translate the question to
        Spanish or the answer to the original language of the question.

        Args:
            state (dict): The current state of the agent, including all keys.

        Returns:
            (bool): Whether or not we have to translate
        """
        logger.debug('Check whether translation is needed')
        state_dict = state["keys"]
        language = state_dict["language"]
        if language != "español":
            translate = True
        else:
            translate = False
        logger.debug("detected language: %s", language)
        return translate

    def answer_question(self, question: str) -> str:
        """
        Invoke RAG-powered chatbot by asking a question.

        Args:
            question (str): Asked question.

        Returns:
            str: Answer to question.
        """
        logger.debug(f'Answering question: {question}')
        inputs = {
            "keys": {
                "question": question,
            },
        }
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                pass
        answer = value['keys']['answer']
        in_quotes = re.match("\\[(.*)\\]", answer)
        if in_quotes:
            answer = in_quotes[1]
        return answer

    def decide_context(self, question: str) -> bool:
        """
        Invoke context detection steps and return whether a given question
        belongs or not to the refugees context.

        Args:
            question (str): Asked question.

        Returns:
            bool: Whether the question belongs or not to the refugees context.
        """
        logger.debug(
            "Determining if the question: %s belongs to the refugees context",
            question)

        # Redefine graph
        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                keys: A dictionary where each key is a string.
            """
            keys: Dict[str, any]

        # Define nodes
        workflow = StateGraph(GraphState)
        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("translate_question", self.translate_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("detect_context", self.detect_context_node)
        # Define flow
        workflow.set_entry_point("detect_language")
        workflow.add_conditional_edges(
            "detect_language",
            self.decide_to_translate,
            {
                True: "translate_question",
                False: "retrieve",
            },
        )
        workflow.add_edge("translate_question", "retrieve")
        workflow.add_edge("retrieve", "detect_context")
        workflow.add_edge("detect_context", END)
        self.graph = workflow.compile()

        inputs = {
            "keys": {
                "question": question,
            },
        }
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                pass
        answer = value['keys']['is_context']
        return answer
