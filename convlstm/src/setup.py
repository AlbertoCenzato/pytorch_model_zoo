from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='convlstmcpp',
      ext_modules=[CppExtension('convlstmcpp', ['convlstm.cpp'])],
      cmdclass={'build_ext': BuildExtension})

#setup(name='lltmcuda',
#      ext_modules=[CUDAExtension('lltmcuda', ['lltm.cu'])],
#      cmdclass={'build_ext': BuildExtension})
