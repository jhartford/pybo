NAME = 'pybo'
VERSION = '0.1.dev3'
AUTHOR = 'Matthew W. Hoffman'
AUTHOR_EMAIL = 'mwh30@cam.ac.uk'


def setup_package(parent_package='', top_path=None):
    from setuptools import find_packages
    from numpy.distutils.core import setup, Extension
    import os, subprocess

    class SimpleExtension(Extension):
        def __init__(self, *sources):
            psources = []
            for source in sources:
                name, ext = os.path.splitext(source)
                if ext == '.pyx':
                    subprocess.call(['cython', source])
                    psources.append(name + '.c')
                else:
                    psources.append(source)

            name, ext = os.path.splitext(psources[0])
            name = name.replace(os.path.sep, '.')

            Extension.__init__(self, name, sources=sources)

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=find_packages(),
        zip_safe=False,
        ext_modules=[
            SimpleExtension('pybo/policies/gpopt/_direct.pyx'),
            SimpleExtension('pybo/policies/gpopt/_escov.pyx',
                            'pybo/policies/gpopt/_escovraw.c'),
        ])


if __name__ == '__main__':
    setup_package()
