from typing import List

def find_vector(folders_2_unzip: List[str]) -> List[str]:
    """
    This function finds vector folders from a list of folders that had previously
    been unzipped related to Copernicus EMS.
    
    Args:
      folders_2_unzip (List[str]): list of folders
      
    Returns:
      list of folder names that are vector type.
    """
    return [folder for folder in folders_2_unzip if is_vector_file(folder) and not is_rtp_file(folder)]

def is_rtp_file(folder_2_unzip: str) -> bool:
    """
    This function checks if an rtp type file exists in the folder_2_unzip directory.
    
    Args:
      folder_2_unzip (str): name of directory.
      
    Returns:
      boolean
    """
    return 'rtp' in folder_2_unzip.lower()
    
def is_vector_file(folder_2_unzip: str) -> bool:
    """
    This function checks if an vector type file exists in the folder_2_unzip directory.
    
    Args:
      folder_2_unzip (str): name of directory.
      
    Returns:
      boolean
    """
    return 'vector' in folder_2_unzip.lower()