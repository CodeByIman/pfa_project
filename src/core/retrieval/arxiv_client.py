# from typing import List, Dict, Optional
# from dataclasses import dataclass
# import arxiv


# @dataclass
# class Paper:
# 	id: str
# 	title: str
# 	authors: List[str]
# 	year: Optional[int]
# 	published: str
# 	abstract: str
# 	pdf_url: Optional[str]
# 	entry_url: str


# def search_arxiv(query: str, max_results: int = 20, sort_by: str = 'relevance') -> List[Paper]:
# 	"""Search arXiv and return a list of papers.
# 	Parameters
# 	- sort_by: 'relevance' | 'lastUpdatedDate' | 'submittedDate'
# 	"""
# 	sort_map = {
# 		'relevance': arxiv.SortCriterion.Relevance,
# 		'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
# 		'submittedDate': arxiv.SortCriterion.SubmittedDate,
# 	}
# 	sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
# 	client = arxiv.Client(page_size=50, delay_seconds=3)
# 	search = arxiv.Search(
# 		query=query,
# 		max_results=max_results,
# 		sort_by=sort_criterion,
# 		sort_order=arxiv.SortOrder.Descending,
# 	)
# 	results: List[Paper] = []
# 	for result in client.results(search):
# 		pdf_url = None
# 		for link in result.links:
# 			if getattr(link, 'title', None) == 'pdf' or (getattr(link, 'href', '').endswith('.pdf')):
# 				pdf_url = link.href
# 				break
# 		year = None
# 		try:
# 			year = result.published.year
# 		except Exception:
# 			pass
# 		paper = Paper(
# 			id=result.entry_id,
# 			title=result.title.strip(),
# 			authors=[a.name for a in result.authors],
# 			year=year,
# 			published=str(result.published),
# 			abstract=result.summary.strip(),
# 			pdf_url=pdf_url,
# 			entry_url=result.entry_id,
# 		)
# 		results.append(paper)
# 	return results 




from typing import List, Dict, Optional
from dataclasses import dataclass
import arxiv
import requests
from datetime import datetime


@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    year: Optional[int]
    published: str
    abstract: str
    pdf_url: Optional[str]
    entry_url: str


def search_arxiv(query: str, max_results: int = 20, sort_by: str = 'relevance', api: str = 'arxiv') -> List[Paper]:
    """Search multiple APIs and return a list of papers.
    
    Parameters:
    - query: search terms
    - max_results: maximum number of results to return
    - sort_by: 'relevance' | 'lastUpdatedDate' | 'submittedDate'
    - api: 'arxiv' | 'semantic_scholar' | 'pubmed' | 'crossref'
    """
    if api == 'arxiv':
        return _search_arxiv(query, max_results, sort_by)
    elif api == 'semantic_scholar':
        return _search_semantic_scholar(query, max_results)
    elif api == 'pubmed':
        return _search_pubmed(query, max_results)
    elif api == 'crossref':
        return _search_crossref(query, max_results)
    else:
        raise ValueError(f"Unsupported API: {api}")


def _search_arxiv(query: str, max_results: int, sort_by: str) -> List[Paper]:
    """Original arXiv search implementation."""
    sort_map = {
        'relevance': arxiv.SortCriterion.Relevance,
        'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
        'submittedDate': arxiv.SortCriterion.SubmittedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    client = arxiv.Client(page_size=50, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    results: List[Paper] = []
    for result in client.results(search):
        pdf_url = None
        for link in result.links:
            if getattr(link, 'title', None) == 'pdf' or (getattr(link, 'href', '').endswith('.pdf')):
                pdf_url = link.href
                break
        year = None
        try:
            year = result.published.year
        except Exception:
            pass
        paper = Paper(
            id=result.entry_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            year=year,
            published=str(result.published),
            abstract=result.summary.strip(),
            pdf_url=pdf_url,
            entry_url=result.entry_id,
        )
        results.append(paper)
    return results


def _search_semantic_scholar(query: str, max_results: int) -> List[Paper]:
    """Search Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,
        'limit': min(max_results, 100),  # API limit
        'fields': 'paperId,title,authors,year,publicationDate,abstract,url,openAccessPdf'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('data', []):
            authors = [author.get('name', 'Unknown') for author in item.get('authors', [])]
            pdf_url = None
            if item.get('openAccessPdf') and item['openAccessPdf'].get('url'):
                pdf_url = item['openAccessPdf']['url']
            
            paper = Paper(
                id=item.get('paperId', ''),
                title=item.get('title', '').strip(),
                authors=authors,
                year=item.get('year'),
                published=item.get('publicationDate', ''),
                abstract=item.get('abstract', '').strip() if item.get('abstract') else '',
                pdf_url=pdf_url,
                entry_url=item.get('url', f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}")
            )
            results.append(paper)
        
        return results
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []


def _search_pubmed(query: str, max_results: int) -> List[Paper]:
    """Search PubMed via NCBI E-utilities."""
    # Search for PMIDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        'db': 'pubmed',
        'term': query,
        'retmax': min(max_results, 200),
        'retmode': 'json'
    }
    
    try:
        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        if not pmids:
            return []
        
        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'json'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        fetch_data = fetch_response.json()
        
        results = []
        articles = fetch_data.get('PubmedArticleSet', {}).get('PubmedArticle', [])
        
        for article in articles:
            medline = article.get('MedlineCitation', {})
            pmid = medline.get('PMID', {}).get('$', '')
            
            article_data = medline.get('Article', {})
            title = article_data.get('ArticleTitle', '').strip()
            abstract_data = article_data.get('Abstract', {})
            abstract = abstract_data.get('AbstractText', '') if abstract_data else ''
            if isinstance(abstract, list):
                abstract = ' '.join([str(a) for a in abstract])
            
            # Extract authors
            authors = []
            author_list = article_data.get('AuthorList', {}).get('Author', [])
            if not isinstance(author_list, list):
                author_list = [author_list]
            
            for author in author_list:
                if isinstance(author, dict):
                    last_name = author.get('LastName', '')
                    first_name = author.get('ForeName', '')
                    if last_name and first_name:
                        authors.append(f"{first_name} {last_name}")
                    elif last_name:
                        authors.append(last_name)
            
            # Extract year
            year = None
            date_completed = medline.get('DateCompleted')
            if date_completed and date_completed.get('Year'):
                year = int(date_completed['Year'])
            
            paper = Paper(
                id=pmid,
                title=title,
                authors=authors,
                year=year,
                published=str(date_completed) if date_completed else '',
                abstract=str(abstract).strip(),
                pdf_url=None,  # PubMed doesn't provide direct PDF links
                entry_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            )
            results.append(paper)
        
        return results
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []


def _search_crossref(query: str, max_results: int) -> List[Paper]:
    """Search CrossRef API."""
    url = "https://api.crossref.org/works"
    params = {
        'query': query,
        'rows': min(max_results, 1000),  # API limit
        'select': 'DOI,title,author,published-print,published-online,abstract,URL'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('message', {}).get('items', []):
            # Extract title
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ''
            
            # Extract authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
            
            # Extract year
            year = None
            published = item.get('published-print') or item.get('published-online')
            if published and published.get('date-parts'):
                year = published['date-parts'][0][0] if published['date-parts'][0] else None
            
            paper = Paper(
                id=item.get('DOI', ''),
                title=title.strip(),
                authors=authors,
                year=year,
                published=str(published) if published else '',
                abstract=item.get('abstract', '').strip(),
                pdf_url=None,  # CrossRef doesn't provide direct PDF links
                entry_url=item.get('URL', f"https://doi.org/{item.get('DOI', '')}")
            )
            results.append(paper)
        
        return results
    except Exception as e:
        print(f"Error searching CrossRef: {e}")
        return []