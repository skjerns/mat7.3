import setuptools

__version__ = None # will be loaded in the next line
exec(open('mat73/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
     name='mat73',  
     version=__version__,
     author="skjerns",
     author_email="nomail@nomail.com",
     description="Load MATLAB .mat 7.3 into Python native data types (via h5/hd5/hdf5/h5py)",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/skjerns/mat7.3",
     download_url=f"https://github.com/skjerns/mat7.3/archive/v{__version__}.tar.gz",
     install_requires=['h5py', 'numpy'],
     tests_require=['scipy'],
     license='MIT',
     packages=['mat73'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

