import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import document_loaders
from langchain_chroma import Chroma

from src.tools.startup import logger
from src.data.load import save_json_file


def execute(settings:dict, global_settings: dict) -> None:
    """
    Index documents and create a vectorstore. This implies:
        - Set embedding model
        - Load the documents
        - Split text of the documents in chunks
        - Create vectorstore and store it in a persistent directory.
    
    Args:
        settings: Execution specific settings.
        global_settings: Global settings.
    """
    logger.debug("Indexing documents")

    collection = settings["collection_name"]
    model_name = settings["model_name"]
    text_splitter = settings["text_splitter"]

    # Setup model
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # extract vectorstore settings for the specified collection
    loader = document_loaders.DirectoryLoader(**settings["documents"])
    docs = loader.load()

    # Define splitting
    split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        **text_splitter)
    all_splits = split.split_documents(docs)

    # store configs used to create the vectorstore
    vectorstore_persist_directory = settings["persist_directory"]
    vectorstore_configs_file = \
        f'{vectorstore_persist_directory}_{collection}_configs.json'

    if os.path.exists(vectorstore_configs_file):
        logger.debug("Deleting old collection \"%s\" in the vectorstore stored "
                     "at \"%s\"", collection, vectorstore_persist_directory)
        client = chromadb.PersistentClient(path=vectorstore_persist_directory)
        try:
            client.delete_collection(name=collection)
        except ValueError:
            pass

    # Set up vector store
    _ = Chroma.from_documents(
        persist_directory=vectorstore_persist_directory,
        documents=all_splits,
        collection_name=collection,
        embedding=embedding_model,
    )

    save_json_file(settings, vectorstore_configs_file)

    logger.debug("Vectorstore ready!")
