import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List

import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Baixar recursos do NLTK, se necessário
for resource in [
    'punkt',
    'punkt_tab',
    'stopwords',
    'gutenberg',
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# Namespace RDF usado pelo Project Gutenberg
NS = {
    'pgterms': 'http://www.gutenberg.org/2009/pgterms/',
    'dcterms': 'http://purl.org/dc/terms/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
}

# Dois livros do Project Gutenberg
BOOKS = [
    {'id': 11, 'label': 'Alice no País das Maravilhas'},
    {'id': 84, 'label': 'Frankenstein'},
]


def fetch_text(book_id: int) -> str:
    urls = [
        f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt',
        f'https://www.gutenberg.org/files/{book_id}/{book_id}.txt',
        f'https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8',
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            if response.ok and response.text.strip():
                return response.text
        except requests.RequestException:
            continue
    raise RuntimeError(f'Não foi possível baixar o texto do livro {book_id}.')


def fetch_rdf(book_id: int) -> str:
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.rdf'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_metadata(rdf_xml: str) -> Dict[str, object]:
    root = ET.fromstring(rdf_xml)
    ebook = root.find('.//pgterms:ebook', NS)
    if ebook is None:
        raise RuntimeError('Metadados RDF não encontrados.')

    title = ebook.findtext('dcterms:title', default='N/D', namespaces=NS)

    authors = []
    for creator in ebook.findall('dcterms:creator', NS):
        agent = creator.find('.//pgterms:agent', NS)
        if agent is not None:
            name = agent.findtext('pgterms:name', default='', namespaces=NS).strip()
            if name:
                authors.append(name)

    languages = []
    for lang in ebook.findall('dcterms:language', NS):
        value = lang.findtext('.//rdf:value', default='', namespaces=NS).strip()
        if value:
            languages.append(value)

    subjects = []
    for subject in ebook.findall('dcterms:subject', NS):
        value = subject.findtext('.//rdf:value', default='', namespaces=NS).strip()
        if value:
            subjects.append(value)

    release_date = ebook.findtext('dcterms:issued', default='N/D', namespaces=NS)
    rights = ebook.findtext('dcterms:rights', default='N/D', namespaces=NS)

    return {
        'title': title,
        'authors': authors,
        'languages': languages,
        'subjects': subjects,
        'release_date': release_date,
        'rights': rights,
    }


def strip_gutenberg_boilerplate(text: str) -> str:
    start_pattern = r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*'
    end_pattern = r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*'

    start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    end_match = re.search(end_pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
    return text.strip()


def analyze_text(text: str) -> Dict[str, object]:
    clean_text = strip_gutenberg_boilerplate(text)

    sentences = sent_tokenize(clean_text, language='english')
    tokens = word_tokenize(clean_text, language='english')

    words_only = [t.lower() for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words_only if w not in stop_words]

    freq = Counter(content_words)
    lexical_diversity = len(set(words_only)) / len(words_only) if words_only else 0

    return {
        'num_sentences': len(sentences),
        'num_tokens': len(tokens),
        'num_words': len(words_only),
        'num_unique_words': len(set(words_only)),
        'lexical_diversity': lexical_diversity,
        'top_words': freq.most_common(10),
    }


def main() -> None:
    print('=' * 80)
    print('ANÁLISE DE CORPORA DO PROJECT GUTENBERG')
    print('=' * 80)

    for book in BOOKS:
        book_id = book['id']
        print(f'\nLivro ID {book_id}')
        print('-' * 80)

        rdf_xml = fetch_rdf(book_id)
        metadata = parse_metadata(rdf_xml)
        text = fetch_text(book_id)
        analysis = analyze_text(text)

        print(f"Título: {metadata['title']}")
        print(f"Autor(es): {', '.join(metadata['authors']) if metadata['authors'] else 'N/D'}")
        print(f"Idioma(s): {', '.join(metadata['languages']) if metadata['languages'] else 'N/D'}")
        print(f"Data de publicação no Project Gutenberg: {metadata['release_date']}")
        print(f"Direitos: {metadata['rights']}")
        print('Assuntos (subjects):')
        for subject in metadata['subjects'][:5]:
            print(f'  - {subject}')

        print('\nEstatísticas textuais:')
        print(f"  - Sentenças: {analysis['num_sentences']}")
        print(f"  - Tokens: {analysis['num_tokens']}")
        print(f"  - Palavras alfabéticas: {analysis['num_words']}")
        print(f"  - Palavras distintas: {analysis['num_unique_words']}")
        print(f"  - Diversidade lexical: {analysis['lexical_diversity']:.4f}")
        print('  - Top 10 palavras de conteúdo:')
        for word, count in analysis['top_words']:
            print(f'      {word}: {count}')

    print('\n' + '=' * 80)
    print('Interpretação prática para PLN:')
    print('- Os textos são os dados linguísticos do corpus.')
    print('- Os metadados permitem filtrar por autor, idioma, assunto e data de publicação no Gutenberg.')
    print('- Isso é útil para montar corpora temáticos, comparar estilos, classificar documentos e fazer buscas semânticas.')
    print('=' * 80)


if __name__ == '__main__':
    main()
