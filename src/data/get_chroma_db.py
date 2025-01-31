import os
import shutil


AWS_DEPLOYMENT = bool(os.environ.get("AWS_DEPLOYMENT", False))


def get_runtime_chroma_path(persist_directory: str) -> str:
    """
    Returns the runtime path for the Chroma database based on whether the 
    code is running in an image-based runtime or not. If `IS_USING_IMAGE_RUNTIME` 
    is set to `True`, the path will be `/tmp/{persist_directory}`, otherwise 
    it remains as the provided `persist_directory`.

    Args:
        persist_directory (str): The path where the Chroma database is
        originally stored.

    Returns:
        str: The adjusted path for the Chroma database to be used at runtime.
    """
    if AWS_DEPLOYMENT:
        return f"/tmp/{persist_directory}"
    else:
        return persist_directory

def copy_chroma_to_tmp(persist_directory: str) -> None:
    """
    Copies the Chroma database from the original `persist_directory` to the 
    temporary path (`/tmp/...`) if it does not already exist there. This is 
    necessary when running in a Lambda runtime that does not have write access
    to the original path.

    Args:
        persist_directory (str): The path where the Chroma database is
        originally stored.
    """
    dst_chroma_path = get_runtime_chroma_path(persist_directory)

    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        print(f"Copying ChromaDB from {persist_directory} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(persist_directory, dst_chroma_path, dirs_exist_ok=True)
    else:
        print(f"✅ ChromaDB already exists in {dst_chroma_path}")


def get_chroma_db(persist_directory: str) -> str:
    """
    Prepares and returns the runtime directory path for the Chroma database. 
    If running in a Lambda runtime, it ensures the Chroma data is available in
    the temporary directory by calling `copy_chroma_to_tmp`.

    Args:
        persist_directory (str): The path where the Chroma database is
        originally stored.

    Returns:
        str: The final path to the Chroma database to be used in the current
        runtime.
    """
    if AWS_DEPLOYMENT:
        copy_chroma_to_tmp(persist_directory)

    persist_directory=get_runtime_chroma_path(persist_directory)
    return persist_directory
