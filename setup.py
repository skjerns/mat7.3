import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
     name='mat73',  
     version='0.3',
     scripts=['__init__.py', 'create_mat.m'] ,
     author="skjerns",
     author_email="nomail",
     description="Load MATLAB .mat 7.3 into Python native data types",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/skjerns/mat7.3",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )