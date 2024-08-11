# Importing find_packages and setup from setuptools for packaging the project
from setuptools import find_packages, setup

# Importing List from typing for type hinting
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    Reads a requirements file and returns a list of requirements.

    This function reads the specified requirements file, processes each line to remove
    newline characters, and returns a list of requirements. If the special requirement
    '-e .' is present in the list, it is removed before returning the list.

    Parameters:
    file_path (str): The path to the requirements file.

    Returns:
    List[str]: A list of requirements as strings.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name='personal_document_masking',
version='0.0.1',
author='CU_23RSG33CU',
author_email='ys2002github@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)