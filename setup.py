import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
     name='mat73',  
     version='0.52',
     author="skjerns",
     author_email="nomail@nomail.com",
     description="Load MATLAB .mat 7.3 into Python native data types",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/skjerns/mat7.3",
     download_url="https://github.com/skjerns/mat7.3/archive/v0.52.tar.gz",
     install_requires=['h5py', 'numpy'],
     license='MIT',
     packages=['mat73'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )