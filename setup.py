from setuptools import setup

setup(
    name='docsearch',
    version='1.0',
    packages=['docsearch'],
    url='https://github.com/ankushbhatia2/docsearch',
    download_url="https://github.com/ankushbhatia2/docsearch/archive/1.0.tar.gz",
    keywords=["information retrieval", "document similarity", "lda", "nlp", "earth movers distance", "jenson shannon"],
    license='Apache License 2.0',
    author='Ankush Bhatia',
    author_email='ankushbhatia02@gmail.com',
    install_requires=['scipy', 'stop_words', 'nltk', 'gensim', 'pyemd', 'pandas'],
    description='Python Module for searching similar documents from a large corpus.'
)
