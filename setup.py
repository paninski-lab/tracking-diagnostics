from distutils.core import setup

setup(
    name='tracking-diagnostics',
    version='0.0.0',
    description='a collection of tools for diagnosing issues in pose estimation',
    author='paninski lab',
    author_email='',
    url='http://www.github.com/paninski-lab/tracking-diagnostics',
    # install_requires=[
    #     'numpy', 'matplotlib', 'sklearn', 'scipy==1.1.0', 'jupyter', 'seaborn'],
    packages=['diagnostics'],
)
