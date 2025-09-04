from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != HYPHEN_E_DOT]

    return requirements



setup(
name="my_first_project",
version = "0.0.1",
author="Abhaykanwar Singh",
author_email="abhaykanwar962@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt"),
long_description=open('README.md').read(),
long_description_content_type="text/markdown"
)