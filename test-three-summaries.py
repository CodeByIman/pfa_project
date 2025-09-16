#!/usr/bin/env python3
"""
Test script for the three summarization methods: TF-IDF, LSA, and Abstractive
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_research_agent.src.core.generation.tfidf_summarizer import summarize_tfidf
from ai_research_agent.src.core.generation.lsa_summarizer import summarize_lsa
from ai_research_agent.src.core.generation.abstractive_summarizer import summarize_abstractive, get_available_models

def test_summarization_methods():
    """Test all three summarization methods with sample text."""
    
    # Sample scientific abstract
    sample_text = """
    Characterizing health informatics journals by subject-level dependencies: a citation
network analysis
Arezo Bodaghi
School of Industrial and Systems Engineering,
Tarbiat Modares University, Iran
arezo.bodaghi@modares.ac.ir
Didi Surian
Centre for Health Informatics, Australian Institute of Health Innovation,
Macquarie University, Australia
1
ABSTRACT
Citation network analysis has become one of methods to study how scientific knowledge flows
from one domain to another. Health informatics is a multidisciplinary field that includes social
science, software engineering, behavioral science, medical science and others. In this study, we
perform an analysis of citation statistics from health informatics journals using data set extracted
from CrossRef. For each health informatics journal, we extract the number of citations from/to
studies related to computer science, medicine/clinical medicine and other fields, including the
number of self-citations from the health informatics journal. With a similar number of articles
used in our analysis, we show that the Journal of the American Medical Informatics Association
(JAMIA) has more in-citations than the Journal of Medical Internet Research (JMIR); while
JMIR has a higher number of out-citations and self-citations. We also show that JMIR cites more
articles from health informatics journals and medicine related journals. In addition, the Journal of
Medical Systems (JMS) cites more articles from computer science journals compared with other
health informatics journals included in our analysis.
Keywords: Citation; citation statistics; health informatics journals
2
INTRODUCTION
Bibliometrics was developed to characterise and understand the inter-connectedness of large
volumes of published research using statistical methods [1]. Citation analyses are a common
method used in bibliometric research and cover studies that examine how authors reference prior
literature, how citations correspond to the characteristics of the research, and the network
structure of citation networks [2]. Health informatics is defined as a study of information and
communication systems in healthcare [3]. Health informatics is a scientific discipline that
handles the intersection of information science, medical informatics, computer science, and
health care informatics [4].
Journals have important differences due to the existence of many research disciplines [5]. These
differences are attributed to intrinsic characteristics of journal. The exchange of citations among
journals forming their positions in a social structure which affect their influence[6]. Our aim was
to characterize the citation structure of health informatics journals to measure differences and
similarities in research focus, the coordination of research across the journals, and differences in
the way the journals are informed by, and inform, medicine and computer science.
RELATED WORK
Networks of collaboration have been investigated extensively using the network science
techniques. The analysis of citation network is performed at three levels including node-level,
group-level, and network-level. The node-level analysis measures the centrality of a node
comprising degree, eigenvector, closeness, and betweenness [7]; the group-level analysis
involves methods for detecting clusters [8]; and the network-level analysis focuses properties of
networks such as distribution of node degrees [9].
3
A wide array of studies have considered the journal citation networks with regard to structural
characteristics such as density, average and largest node distances, percolation robustness,
distributions of incoming and outgoing edges, reciprocity, and assortative mixing by node
degrees [10]. There are studies in which journal citation networks were analyzed empirically and
focusing on communities in citation networks [11, 12]. However, most of previous studies only
focused on a specific journal in the analysis.
MATERIAL AND METHODS
Study data
We selected the first ten health informatics journals ranked by Google Scholar [13] in the
“medical informatics” sub-discipline. We identified 10,716 articles published in the top five
health informatics journals from 1944 to 2018. From the 10,716 articles, the reference lists were
available for about 1,944 articles. The information of the five health informatics journals
including the digital object identifiers (DOIs) for all 1944 articles, and reference lists with the
DOIs, journals’ ISSN, and name for all cited references were retrieved from CrossRef
(https://www.crossref.org). All journals extracted from the reference lists were labelled using
CrossRef’s subject list and abstracted to one of four different groups: health informatics,
medicine, computer science, and others. Although some journals were listed in one subject
category, the others were listed in multiple subject categories. For those journals with multiple
subjects, we manually assigned them to the most relevant subject category. Currently, there are
some journals that information about references and citations to CrossRef are not provided,
whereas they might appear among the reference lists of articles published in journals that were
included in the analysis.
4
Network construction
We generated a journal citation network from the main health informatics journals and the other
extracted journals. Each journal is represented by a node and the relation between two journals is
represented by an edge (a directed edge goes from an article to the article in its reference list).
This network is a directed graph with 4,144 nodes (journals) and 39,656 edges among journals.
Furthermore, we constructed another directed network of citations exchange among main papers
for which reference lists were harvested. In this network, all 1,944 papers are considered as
nodes, and edges are directed links between papers. The third network was a bipartite network
comprising two types of nodes (journals and subjects) in which all five health informatics
journals are on the left side, and four different subjects on the right side. In the bipartite network,
there is an edge when a health informatics journal cites to a journal from a subject. Figure 1
illustrates the citation network among the journals. We used winpython with networkx and igraph
libraries in our experiments. To construct and visualize the citation network, we used Gephi.
5
Figure 1: The relation among journals in four different groups: health informatics, medicine,
computer science, and others.
Analyses
In our study, journals’ overall attractiveness is measured with several measures containing
incoming and outgoing citations, followed by the number of out-citations in different subjects
(network and group level analysis). For investigating the role of aforementioned factors in tie-
generation in the directed network of citations, various statistical terms associated with them
namely in in-degree, out-degree, and loops were considered in this study. Using the number of
different citations we can find out which health informatics journal receive most or least citations
per paper from other journals (in-citation), and the journal that has more citations per paper to
6
other journals (out-citations). Moreover, we can identify the health informatics journals with the
highest number of citations to its papers (self-citation).
In terms of relations across subjects, we investigated the behavior of five health informatics
journals (Journal of Medical Internet Research; Journal of the American Medical Informatics
Association; Journal of Medical Systems; BMC Medical Informatics and Decision Making;
Journal of Medical Internet Research - Mobile Health and Ubiquitous Health) in the citations to
other journals of different subjects (health informatics, computer science, medicine, other fields).
The number of out-citations in every different subject indicates the degree of dependence or
application of different subject in health informatics.
Results
The section presents results of the survey. The data available for these journals varied in terms of
the years for which articles were available and the years in which articles had reference list data
available is shown in Table 1. In addition, the relation among main articles that the list of
references are available for them, is demonstrated as a network in Figure 2.
Table 1: The main health informatics journals extracted from CrossRef
Health informatics journals Available Number of Number of
publication available articles with
years articles reference data
Journal of Medical Internet Research (JMIR) 1999- 2018 2779 525
Journal of the American Medical Informatics 1994- 2018 3021 524
Association (JAMIA)
Journal of Medical Systems (JMS) 1977- 2018 2843 470
BMC Medical Informatics and Decision Making 2001- 2018 1410 287
(BMC MIDM)
Journal of Medical Internet Research - Mobile Health 2013- 2018 663 138
and Ubiquitous Health (JMU)
7
Figure 2: The directed network of 1,944 papers from five health informatics journals (JAMIA:
blue; JMS; green; MIDM: red; JMIR: purple; JMU: dark green). Node sizes are proportional to
the number of incoming citations. In this network the JAMIA cluster clearly is close to JMIR,
while BMC MIDM is placed opposite to JAMIA.
The characteristics of the constructed citation network is shown in Table 2 and the citation
network from the five main health informatics journals to the other journals is illustrated by
Figure 3. All the main health informatics journals are positioned on the left side and the rest
nodes on the right side.
8
    """
    
    print("🧪 Testing Three Summarization Methods")
    print("=" * 60)
    print(f"📝 Input text length: {len(sample_text)} characters")
    print()
    
    # Test TF-IDF
    print("1️⃣ TF-IDF Summarization:")
    print("-" * 30)
    try:
        tfidf_summary = summarize_tfidf(sample_text, max_sentences=3)
        print(f"✅ TF-IDF Summary ({len(tfidf_summary)} chars):")
        print(f"   {tfidf_summary}")
    except Exception as e:
        print(f"❌ TF-IDF failed: {e}")
    print()
    
    # Test LSA
    print("2️⃣ LSA Summarization:")
    print("-" * 30)
    try:
        lsa_summary = summarize_lsa(sample_text, max_sentences=3)
        print(f"✅ LSA Summary ({len(lsa_summary)} chars):")
        print(f"   {lsa_summary}")
    except Exception as e:
        print(f"❌ LSA failed: {e}")
    print()
    
    # Test Abstractive
    print("3️⃣ Abstractive Summarization:")
    print("-" * 30)
    try:
        abstractive_summary = summarize_abstractive(sample_text, max_length=150, min_length=40)
        print(f"✅ Abstractive Summary ({len(abstractive_summary)} chars):")
        print(f"   {abstractive_summary}")
    except Exception as e:
        print(f"❌ Abstractive failed: {e}")
    print()
    
    # Show available models
    print("🤖 Available Abstractive Models:")
    print("-" * 30)
    models = get_available_models()
    if models:
        for model in models:
            print(f"   • {model}")
    else:
        print("   No models available (transformers not installed)")
    print()
    
    # Test pipeline integration
    print("🔄 Testing Pipeline Integration:")
    print("-" * 30)
    try:
        from ai_research_agent.src.core.agent.orchestrator import run_pipeline
        
        print("Running full pipeline test...")
        result = run_pipeline(
            query="deep learning medical imaging",
            max_results=2,
            max_pdfs=1,
            top_k=1,
            api="arxiv",
            use_pdfs=False
        )
        
        if result['results']:
            paper = result['results'][0]
            print(f"✅ Pipeline test successful!")
            print(f"   Paper: {paper['title'][:60]}...")
            print(f"   Score: {paper['score']:.3f}")
            print(f"   Has summaries: {'summaries' in paper}")
        else:
            print("❌ No results from pipeline")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
    
    print()
    print("🎉 Test completed!")

if __name__ == "__main__":
    test_summarization_methods()
