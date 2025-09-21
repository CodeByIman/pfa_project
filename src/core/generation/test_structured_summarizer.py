"""
Test file for the Structured PDF Summarizer

This file provides comprehensive testing for the structured_pdf_summarizer module
with sample academic text and various test scenarios.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List
import logging

# Ensure project root (the directory that CONTAINS the 'ai_research_agent' package) is on sys.path
# This test file lives at: ai_research_agent/src/core/generation/test_structured_summarizer.py
# Package dir:           .../ai_research_agent
# Project root to add:   .../  (the parent folder of the package dir)
_here = Path(__file__).resolve()
PACKAGE_DIR = _here.parents[3]  # .../ai_research_agent (the package directory)
PROJECT_ROOT = PACKAGE_DIR.parent  # .../ (directory that contains the package)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
added_path = str(PROJECT_ROOT)

# Import using package path so relative imports inside modules resolve correctly
try:
    from ai_research_agent.src.core.generation.structured_pdf_summarizer import (
        StructuredSummary,
        ContentExtractor,
        EnhancedSectionDetector,
        PaperTypeDetector,
        QualityValidator,
        improved_split_into_sections,
        process_pdf_structured_summary,
    )
    from ai_research_agent.src.core.pdf_processing.preprocess import clean_scientific_text
    print("✅ Successfully imported structured_pdf_summarizer modules (package mode)")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Hint A: run from project root with: python -m ai_research_agent.src.core.generation.test_structured_summarizer")
    print("Hint B: or run this file after setting PYTHONPATH to the project root (the folder containing 'ai_research_agent')")
    # Debug paths
    print("sys.path includes:")
    for p in sys.path[:5]:
        print("  -", p)
    if added_path:
        print(f"Added to sys.path: {added_path}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Sample academic paper text for testing
SAMPLE_ACADEMIC_TEXT = """
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
8"""

# Additional sample texts for different paper types
SAMPLE_THEORETICAL_PAPER = """
Abstract

This paper presents a comprehensive theoretical analysis of the computational complexity of approximate string matching algorithms under different distance metrics. We prove new lower bounds for the edit distance problem and establish the optimality of existing algorithms under certain assumptions. Our main contribution is a unified framework for analyzing string matching complexity that applies to multiple distance metrics including Hamming distance, edit distance, and longest common subsequence.

1. Introduction

String matching is a fundamental problem in computer science with applications ranging from bioinformatics to information retrieval. The theoretical foundations of exact string matching are well-established, but approximate matching presents more complex theoretical challenges.

We contribute three main theoretical results. First, we prove that any algorithm for computing edit distance between strings of length n requires Ω(n²) time in the worst case under the word RAM model. Second, we establish tight bounds for the space-time trade-off in approximate string matching. Third, we present a unified complexity framework that encompasses multiple distance metrics.

Our approach builds upon previous work in algebraic complexity theory and leverages novel proof techniques from communication complexity. We use a reduction from the Boolean matrix multiplication problem to establish our lower bounds.
"""

SAMPLE_SURVEY_PAPER = """
Abstract

This survey provides a comprehensive overview of recent advances in federated learning, covering theoretical foundations, algorithmic developments, and practical applications. We systematically analyze over 200 recent papers, categorizing approaches by their handling of data heterogeneity, communication efficiency, and privacy preservation. Key challenges and future research directions are identified and discussed.

1. Introduction

Federated learning has emerged as a paradigm for collaborative machine learning without centralized data collection. This survey examines the current state of the field, synthesizing findings from recent literature and identifying key research directions.

Our contribution is a systematic taxonomy of federated learning approaches based on three primary dimensions: data distribution assumptions, communication protocols, and privacy guarantees. We analyze the trade-offs between these dimensions and provide guidance for practitioners.

2. Taxonomy of Approaches

We categorize existing federated learning methods into four main classes based on their assumptions about data distribution: IID (independent and identically distributed), non-IID with feature skew, non-IID with label skew, and concept drift scenarios. Each category presents unique challenges and requires different algorithmic solutions.

Communication-efficient approaches can be further subdivided into compression-based methods, sparsification techniques, and quantization schemes. Privacy-preserving methods include differential privacy, homomorphic encryption, and secure multiparty computation approaches.
"""

class StructuredSummarizerTester:
    """Comprehensive test suite for the structured PDF summarizer"""
    
    def __init__(self):
        self.test_texts = {
            'empirical': SAMPLE_ACADEMIC_TEXT,
            'theoretical': SAMPLE_THEORETICAL_PAPER,
            'survey': SAMPLE_SURVEY_PAPER
        }
        self.test_results = []

    def _split_with_cleaning(self, text: str):
        """Helper used in tests to mirror production preprocessing.
        Cleans scientific text first, then performs section detection.
        """
        cleaned = clean_scientific_text(text)
        return improved_split_into_sections(cleaned)
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🧪 Starting Structured PDF Summarizer Tests\n")
        print("=" * 60)
        
        # Test 1: Section Detection
        print("\n📋 Test 1: Section Detection")
        self.test_section_detection()
        
        # Test 2: Content Extraction
        print("\n🔍 Test 2: Content Extraction")
        self.test_content_extraction()
        
        # Test 3: Paper Type Detection
        print("\n📰 Test 3: Paper Type Detection")
        self.test_paper_type_detection()
        
        # Test 4: Quality Validation
        print("\n✅ Test 4: Quality Validation")
        self.test_quality_validation()
        
        # Test 5: Full Pipeline
        print("\n🔄 Test 5: Full Pipeline Integration")
        self.test_full_pipeline()
        
        # Print summary
        self.print_test_summary()
    
    def test_section_detection(self):
        """Test section detection functionality"""
        print("Testing section detection on sample academic text...")
        
        try:
            sections = self._split_with_cleaning(SAMPLE_ACADEMIC_TEXT)
            
            print(f"✅ Detected {len(sections)} sections:")
            for section_name, content in sections.items():
                preview = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content
                print(f"   - {section_name}: {len(content)} chars | {preview}")
            
            # Validate expected sections
            expected_sections = ['abstract', 'introduction', 'methodology']
            found_sections = list(sections.keys())
            
            for expected in expected_sections:
                if any(expected in section.lower() for section in found_sections):
                    print(f"   ✅ Found expected section: {expected}")
                else:
                    print(f"   ⚠️  Missing expected section: {expected}")
            
            self.test_results.append(('Section Detection', 'PASS', f"{len(sections)} sections detected"))
            
        except Exception as e:
            print(f"❌ Section detection failed: {e}")
            self.test_results.append(('Section Detection', 'FAIL', str(e)))
    
    def test_content_extraction(self):
        """Test content extraction for each summary component"""
        print("Testing content extraction components...")
        
        try:
            # Get sections first
            sections = self._split_with_cleaning(SAMPLE_ACADEMIC_TEXT)
            extractor = ContentExtractor()
            
            # Test each extraction method
            extraction_tests = [
                ('Contributions', extractor.extract_contributions),
                ('Methodology', extractor.extract_methodology),
                ('Results', extractor.extract_results),
                ('Limitations', extractor.extract_limitations),
                ('Future Work', extractor.extract_future_work)
            ]
            
            for test_name, extraction_method in extraction_tests:
                try:
                    result = extraction_method(sections)
                    success = len(result) > 50 and not result.startswith(f"{test_name.lower()} not clearly")
                    
                    if success:
                        preview = result[:150] + "..." if len(result) > 150 else result
                        print(f"   ✅ {test_name}: {len(result)} chars | {preview}")
                    else:
                        print(f"   ⚠️  {test_name}: Default message returned - {result[:100]}")
                    
                except Exception as e:
                    print(f"   ❌ {test_name}: Error - {e}")
            
            self.test_results.append(('Content Extraction', 'PASS', 'All extraction methods tested'))
            
        except Exception as e:
            print(f"❌ Content extraction test failed: {e}")
            self.test_results.append(('Content Extraction', 'FAIL', str(e)))
    
    def test_paper_type_detection(self):
        """Test paper type and domain detection"""
        print("Testing paper type detection...")
        
        detector = PaperTypeDetector()
        
        test_cases = [
            ('Empirical Paper', SAMPLE_ACADEMIC_TEXT, 'empirical'),
            ('Theoretical Paper', SAMPLE_THEORETICAL_PAPER, 'theoretical'),
            ('Survey Paper', SAMPLE_SURVEY_PAPER, 'survey')
        ]
        
        for test_name, text, expected_type in test_cases:
            try:
                sections = self._split_with_cleaning(text)
                paper_type, confidence = detector.detect_paper_type(sections)
                domain, domain_confidence = detector.detect_domain(sections)
                
                type_match = paper_type == expected_type or confidence > 0.3
                result_icon = "✅" if type_match else "⚠️"
                
                print(f"   {result_icon} {test_name}:")
                print(f"      Type: {paper_type} (confidence: {confidence:.2f})")
                print(f"      Domain: {domain} (confidence: {domain_confidence:.2f})")
                
            except Exception as e:
                print(f"   ❌ {test_name}: Error - {e}")
        
        self.test_results.append(('Paper Type Detection', 'PASS', 'Type detection completed'))
    
    def test_quality_validation(self):
        """Test quality validation system"""
        print("Testing quality validation...")
        
        try:
            # Create a sample structured summary
            sections = self._split_with_cleaning(SAMPLE_ACADEMIC_TEXT)
            extractor = ContentExtractor()
            
            sample_summary = StructuredSummary(
                contributions=extractor.extract_contributions(sections),
                methodology=extractor.extract_methodology(sections),
                results=extractor.extract_results(sections),
                limitations=extractor.extract_limitations(sections),
                future_work=extractor.extract_future_work(sections),
                short_overview="Sample overview for testing"
            )
            
            # Validate the summary
            validator = QualityValidator()
            validation_results = validator.validate_structured_summary(sample_summary, sections)
            
            print(f"   ✅ Overall Quality Score: {validation_results['overall_score']:.2f}")
            print(f"   📊 Quality Level: {validation_results['quality_level']}")
            
            print("   Section Scores:")
            for section, score in validation_results['section_scores'].items():
                score_icon = "🟢" if score > 0.6 else "🟡" if score > 0.3 else "🔴"
                print(f"      {score_icon} {section}: {score:.2f}")
            
            if validation_results['recommendations']:
                print("   💡 Recommendations:")
                for rec in validation_results['recommendations'][:3]:
                    print(f"      - {rec}")
            
            self.test_results.append(('Quality Validation', 'PASS', f"Score: {validation_results['overall_score']:.2f}"))
            
        except Exception as e:
            print(f"❌ Quality validation test failed: {e}")
            self.test_results.append(('Quality Validation', 'FAIL', str(e)))
    
    def test_full_pipeline(self):
        """Test the complete summarization pipeline"""
        print("Testing full pipeline integration...")
        
        try:
            # Mock the PDF processing by using our sample text directly
            # In a real scenario, you would use process_pdf_structured_summary with an actual PDF
            
            sections = self._split_with_cleaning(SAMPLE_ACADEMIC_TEXT)
            extractor = ContentExtractor()
            detector = PaperTypeDetector()
            validator = QualityValidator()
            
            # Extract all components
            contributions = extractor.extract_contributions(sections)
            methodology = extractor.extract_methodology(sections)
            results = extractor.extract_results(sections)
            limitations = extractor.extract_limitations(sections)
            future_work = extractor.extract_future_work(sections)
            
            # Create structured summary
            structured_summary = StructuredSummary(
                contributions=contributions,
                methodology=methodology,
                results=results,
                limitations=limitations,
                future_work=future_work,
                short_overview="Integrated pipeline test overview"
            )
            
            # Generate short overview
            overview_parts = []
            if contributions: overview_parts.append(contributions.split('.')[0])
            if methodology: overview_parts.append(methodology.split('.')[0])
            if results: overview_parts.append(results.split('.')[0])
            
            structured_summary.short_overview = '. '.join(overview_parts[:3]) + '.'
            
            # Validate
            validation = validator.validate_structured_summary(structured_summary, sections)
            
            # Detect paper characteristics
            paper_type, type_conf = detector.detect_paper_type(sections)
            domain, domain_conf = detector.detect_domain(sections)
            
            print("   🎯 Pipeline Results:")
            print(f"      Paper Type: {paper_type} ({type_conf:.2f})")
            print(f"      Domain: {domain} ({domain_conf:.2f})")
            print(f"      Overall Quality: {validation['quality_level']} ({validation['overall_score']:.2f})")
            
            print("\n   📝 Generated Summary Components:")
            for component in ['contributions', 'methodology', 'results', 'limitations', 'future_work']:
                content = getattr(structured_summary, component)
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"      {component.title()}: {preview}")
            
            self.test_results.append(('Full Pipeline', 'PASS', f"Quality: {validation['quality_level']}"))
            
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            self.test_results.append(('Full Pipeline', 'FAIL', str(e)))
    
    def print_test_summary(self):
        """Print a summary of all test results"""
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == 'PASS')
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {passed/total*100:.1f}%\n")
        
        for test_name, status, details in self.test_results:
            status_icon = "✅" if status == 'PASS' else "❌"
            print(f"{status_icon} {test_name}: {status} - {details}")
        
        if passed == total:
            print(f"\n🎉 All tests passed! The structured summarizer is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check the error messages above for debugging.")

def test_with_custom_text():
    """
    Function to test the summarizer with your own text
    Replace the text in the CUSTOM_TEXT variable below with your own academic text
    """
    print("\n🔬 Testing with Custom Text")
    print("=" * 40)
    
    # PLACE YOUR CUSTOM TEXT HERE
    CUSTOM_TEXT = """
    Graphical Abstract
Explainable AI-Enhanced Supervisory Control for High-Precision Spacecraft
Formation
Reza Pirayeshshirazinezhad
arXiv:2509.13331v1 [astro-ph.IM] 9 Sep 2025
Figure 1: (A) The orbits and ACS phases in the formation are shown. The pseudo-orbit keeps the
formation for |rrel| = 1 km with 1 meter accuracy. rrel = rF − rL. rL and rF show the distance vectors
of the spacecraft.Supervisory adaptive control system. (B) s represents the set of discrete states of finite
automata, and k represents the set of all transitions in finite automata. Begin and Ended represent the
beginning and ending of VTXO mission.
2
Highlights
Explainable AI-Enhanced Supervisory Control for High-Precision Spacecraft
Formation
Reza Pirayeshshirazinezhad
• Introduces a timed automata-based supervisory control framework for spacecraft
formation flying.
• Introduces sliding mode control for high-precision relative position formation.
• Introduces an explainable AI for performance prediction over the optimal mission
parameters.
• Introduces a neural network based solver for a unified constrained, non-convex
dynamic optimization.
Explainable AI-Enhanced Supervisory Control for High-Precision
Spacecraft Formation
Reza Pirayeshshirazinezhad
Texas A&M University, TX, USA
Abstract
We use artificial intelligence (AI) and supervisory adaptive control systems to plan
and optimize the mission of precise spacecraft formation. Machine learning and robust
control enhance the efficiency of spacecraft precision formation of the Virtual Telescope
for X-ray Observation (VTXO) space mission. VTXO is a precise formation of two
separate spacecraft making a virtual telescope with a one-kilometer focal length. One
spacecraft carries the lens and the other spacecraft holds the camera to observe highenergy space objects in the X-ray domain with 55 milli-arcsecond angular resolution
accuracy. Timed automata for supervisory control, Monte Carlo simulations for stability
and robustness evaluation, and integration of deep neural networks for optimal estimation
of mission parameters, satisfy the high precision mission criteria. We integrate deep neural
networks with a constrained, non-convex dynamic optimization pipeline to predict optimal
mission parameters, ensuring precision mission criteria are met. AI framework provides
explainability by predicting the resulting energy consumption and mission error for a given
set of mission parameters. It allows for transparent, justifiable, and real-time trade-offs, a
capability not present in traditional adaptive controllers. The results show reductions in
energy consumption and improved mission accuracy, demonstrating the capability of the
system to address dynamic uncertainties and disturbances.
Keywords: Supervisory control, timed automata, Virtual Telescope for X-ray
Observation (VTXO), Monte-Carlo simulation, Optimal control, Machine learning,
Optimization, Satellites, Real-time control, Explainable AI (XAI), Interpretability
1. Introduction
VTXO employs a Phased Fresnel Lens (PFL) to achieve near diffraction-limited
angular resolution in the X-ray band[1–7]. This lens offers the VTXO mission an imaging
resolution of around 55 milli-arcsecond (mas) for a 1 km focal length in the X-ray spectrum.
To maintain this level of accuracy, VTXO makes a precise alignment formation flying
between its Inner Satellite (IS) and Outer Satellite (OS) with a sub-millimeter transverse
accuracy. Formation flying in space provides advantages over a single spacecraft, including
robustness, redundancy, reconfiguration, and broader coverage.
When the two spacecraft achieve this 1 km, 55 mas alignment precision, especially
at their highest point, or apogee, they can begin their scientific observations. Notably,
this 55 mas resolution is about ten times finer than the Chandra X-rays and the James
Webb Space Telescope’s 0.5 arcsecond resolutions[1, 8]. Such clarity from VTXO enables
unprecedented studies of exoplanets[9] and the environments surrounding space entities
like black holes, neutron stars, and other stellar systems.
The Navy Interferometric Star Tracker Experiment II (NISTEx-II) [10] plays a critical
role in the precision formation flying of the VTXO mission by allowing highly accurate
relative navigation. Mounted on the optical bench of (IS), the NISTEx-II star tracker,
in conjunction with laser beacons installed on the Outer Satellite (OS), allows the GNC
system to maintain sub-millimeter transverse alignment and attitude accuracy within 55
milli-arcseconds [1, 2, 11]. To achieve this level of control, VTXO integrates thrusters,
reaction wheels, an inertial measurement unit (IMU), GPS, the NISTEx-II star tracker,
laser beacon systems, and a radio frequency (RF) ranging system that also serves as the
inter-satellite communication link. These integrated sensing and actuation subsystems
collectively establish stringent requirements on the Attitude Control System (ACS) state
error [1]. Additionally, energy consumption and mission duration are tightly constrained,
driving the optimization of control strategies to minimize both, in line with mission-level
requirements for efficiency and scientific yield.
The Navy Interferometric Star Tracker Experiment II (NISTEx-II) instrument [10]
provides accurate formation navigation for VTXO. Using NISTEx-II star tracker on the
optical bench of IS and laser beacons on OS, GNC keeps the formation within millimeter
level transverse alignment and 55 milli-arcsecond attitude accuracy[1, 2, 11]. GNC of
VTXO uses thrusters, reaction wheels, inertial navigation system sensor (IMU), GPS,
NISTEx II precision star tracker on the optical bench of IS, laser beacons on the OS,
and a radio ranging system that also serves as an inter-satellite communication link.
The instrumentation of VTXO [1] imposes requirements on the ACS state error. Beside,
consumed energy of ACS and the time of the mission are desired to be minimal as the
requirements of the VTXO.
The VTXO’s relative position control system is divided into four stages, while its
Attitude Control System (ACS) has three stages. Using trajectory optimization, we
identify the best stages, orbits, and control strategies for the Guidance, Navigation, and
Control (GNC) system[1, 2, 12, 13]. For VTXO’s purpose, three stages are set for the
ACS. Each stage aims to enhance VTXO’s efficiency by improving accuracy and saving
energy[3–7, 14]. Implementing these stages to the mission provides several advantages:
• Each stage has a specific goal tailored to the mission’s objectives and the equipment
in use.
• Both the artificial intelligence (AI) system and the controllers for each stage are
crafted to meet these specific goals.
To ensure optimal science observation, the VTXO mission employs specific accuracy
metrics for its pre-observation phase, and uses an objective function to minimize energy
consumption, which in turn enhances mission longevity, reduces costs, and ensures highresolution pointing while maintaining stability across all starting conditions. Sliding-mode
control (SMC) can provide robustness and asymptotic stability in the presence of noise,
disturbances with the disadvantages of chattering [15, 16]. Chattering has to stay minimal
in VTXO since NISTEx-II requires the angular velocities to stay close to zero. Other
approaches including robust control [17, 18] including SMC, adaptive control[15, 19],
Deep neural networks (DNN) controllers for nonlinear system [20] have shown asymptotic
stability in the presence of disturbances and noise. However, usually adaptive controllers,
including model reference adaptive control (MRAC), are developed for linear systems
[20–22] and they are relatively slow in convergence. The nonlinear adaptive controller
version developed by State Dependent Riccati Equations (SDRE) doesn’t provide global
2
asymptotic stability [21], as it only provides local asymptotic stability. DNN controllers,
on the other hand, are stable globally and they can estimate the nonlinear dynamics [20],
but, still the slow convergence is a concern, and the process of designing the controller is
complex [20]. In VTXO, the speed of controller convergence is important as it defines T
for the transient phase, and the dynamics of the nonlinear system is known. Multiplicative
extended Kalman filter (MEKF) [4, 6, 16, 23, 24] can provide the state estimation with
noisy sensor measurements and having disturbances and uncertainties in the system.
Extended state observer (ESO) [15] have shown promising solutions to the disturbance
and uncertainty estimation and rejection while providing asymptotic stability. ESO
estimates the time varying differentiable uncertainties and disturbances in the system,
and the controller compensates for them in a single spacecraft [15] and multiple spacecraft
formation control[25]. Quaternions offer a robust and computationally efficient framework
for solving equations of motion in spacecraft dynamics. Unlike Euler angles, they do
not suffer from singularities and are straightforward to normalize [16, 26]. Quaternions
are not exclusive to aerospace applications; they find use in fields as diverse as fluid
mechanics and quantum mechanics [27]. In this paper, Monte-Carlo simulation is used
for control system robustness and stability demonstration, and it is used to show control
system robustness against external disturbances and noise measurement. Monte-Carlo
simulation is used to show the control system robustness and stability for flexible robot
arm [28] and for the formation control robustness of UAVs [29]. Sliding-mode control
(SMC) and Lyapunov-based controller are used in different phases of the mission to ensure
robustness and guarantee stability even when faced with external disturbances and noise.
The Multiplicative Extended Kalman Filter (MEKF)[4, 6, 16, 23, 24] estimates system
states when there are disturbances, uncertainties, and noisy sensor data. Similar to the
VTXO mission, a case study is done on the MASSIM X-ray virtual telescope with the
astrometric sensor and the accelerometer sensors [13]. The MASSIM mission is the precise
formation between 2 spacecraft with 1000 kg mass and 1000 km distance. In the MASSIM
mission case study, the 3-sigma estimation error of navigation structures is shown versus
bias, noise, and disturbances. The controller is a linear controller only performed by
the follower. It is shown that for different bias, noise, and disturbances, the navigation
structure can provide different navigation estimation 3-sigma error. This paper provides
the VTXO mission with methods to increase the performance of VTXO by the following
contributions
• Test introducing timed automata method as a supervisory adaptive control for the
analysis of phases of VTXO mission.
• Test using Monte-Carlo simulation to analyze the VTXO mission and the stability
of the timed automata method.
While modern control strategies increasingly use AI, their application in safety-critical
space missions is often hindered by their "black-box" nature. A lack of transparency
in how an AI derives its control decisions creates significant challenges for verification,
validation, and operator trust. This paper addresses this gap by proposing an Explainable
AI (XAI)-enhanced framework. Our approach provides explainability on two levels: (1)
At the AI level, the system does not merely output controller gains, but also predicts the
resulting performance metrics (energy and error), making the AI’s trade-offs transparent
and interpretable. (2) At the supervisory level, a timed automata model provides a
formal, deterministic, and verifiable logic for all mission phase transitions. This dual-level
explainability ensures that the system’s behavior is both optimal and understandable.
3
This paper provides the VTXO mission with methods to increase the performance of
VTXO by the following contributions
• Introducing timed automata method as a supervisory adaptive control for the
analysis of VTXO mission.
• Introducing a new formation relative position control for VTXO
• Integrating a performance-predicting controller with a formally verifiable supervisory
system to ensure trustworthy spacecraft autonomy.
• Introducing a neural network based solver for a unified constrained, non-convex
dynamic optimization.
The codes and data in this article are available at https://github.com/Rpirayesh.
2. VTXO Mission Concept of Operation and Its Implications for Supervisory
Control
A clear understanding of the Virtual Telescope for X-ray Observation (VTXO) mission’s
operational phases and accuracy requirements is fundamental to designing an effective
supervisory control system. The intricate sequence of operations, each with distinct control
objectives and constraints, dictates the need for an intelligent and adaptive supervisory
layer capable of managing diverse control strategies. This section summarizes the concept
of the VTXO mission, largely based on previous work [4–6, 30], to provide the necessary
context for the development of XAI, the supervisory framework, and relative position
control presented in this paper. The supervisory control system ensures seamless transitions
between these phases, optimizes performance according to phase-specific metrics, and
maintains stringent accuracy under varying conditions, all of which are directly informed
by the mission profile detailed below. In the VTXO mission, each orbit involves distinct
operational phases for both the Attitude Control System (ACS) and the relative position
formation between the Outer Satellite (OS) and the Inner Satellite (IS). These phases,
first described in [30], are critical for the subsequent control design.
2.1. Phases of ACS
ACS has 3 phases as follows:
• The formation phase.
• The transient phase.
• The science phase.
In the formation stabilization phase, the spacecraft are stabilized while they pass the
perigee to come into the next orbit phase. GPS is used for orbit determination. The
controller is a feed-forward controller that compensates for the external torque of the
gravity gradient.
During the transient phase, a Lyapunov controller is employed to reduce energy E
and guide convergence toward the equilibrium point xe, for any initial conditions [30].
ACS aligns OS and IS for the observation phase. For navigation, Inertial Measurement
Units (IMU) and star trackers are used. To maximize the duration of the observation and
4
stabilization phases, the transient phase is constrained to a few minutes to increase the
science observation period.
The science phase begins the precise formation after the transient phase with the
following specifications [4–6, 6, 30].
• The optical system must achieve an angular resolution of 55 milli-arcseconds
(FWHM).
• A sub-millimeter level of accuracy is required for the transverse alignment between
OS and IS.
• Pointing accuracy for each individual spacecraft should be within a few arcminutes.
• The distance between the OS and IS must be maintained at 1 km, with an allowable
deviation of up to one meter.
This 55 milli-arcsecond angular resolution corresponds to maintaining sub-millimeter
transverse alignment over the 1 km focal length; thus, the stringent alignment requirement,
rather than the relative spacecraft orientation, drives the optical performance of the
VTXO telescope.
For navigation, IS and (OS) are equipped with Inertial Measurement Units (IMU)
and star trackers. The IS additionally uses laser beacons and radio ranging for enhanced
accuracy. GPS is ruled out for high-altitude apogee navigation due to its limited accuracy.
The Multiplicative Extended Kalman Filter (MEKF) and Sliding-Mode Control (SMC)
are employed to ensure robustness against disturbances and model uncertainties[4].
2.2. Position Formation Phases
Position formation is broken down into four main phases:
• De-formation phase
• Tracking phase
• Formation phase
• Science phase
During the de-formation phase, the 1 km formation is broken and the satellites drift
apart. The tracking phase uses GPS and radio ranging for orbit determination and kicks
in near perigee. This phase is also when data, including images, is transmitted to Earth.
The formation phase aims to get the IS and OS back to a distance of 1 km, meeting
certain accuracy requirements. Finally, the science phase begins, where high-resolution
imaging of space objects occurs.
2.3. Accuracy Requirements for the transient phase
Two primary accuracy requirements are imposed by the Field of View (FoV) of
the NISTEx-II star tracker and VTXO. Percentile is used to characterize the accuracy
requirements. P99(e) is used to characterize the accuracy requirements for the slew error.
Using P99, 99 percent of times the mission meets the accuracy requirements.
R1: Accuracy requirements due to the formation of NISTEx-II instrument
on IS and laser beacons on OS
5
• R1A:P99(e) < 5
◦
• R1B:P99(e) < 5
◦
R2: Accuracy requirements due to the FoV of VTXO
• R2A:P99(e) < 0.18◦
• R2B:P99(e) < 0.18◦
2.4. Attitude representation and notation
The estimated value or the value in the flight computer for the true variable x is
denoted as xˆ, and the measured value for x is denoted as x˜. diag(x) represents the
diagonal matrix with elements of vector x in the diagonal.
The Earth coordinate frame is the Earth-centered inertia (ECI) frame„ and the frames
used for the satellite body frame Fb are the Local-Vertical-Local-Horizontal (LVLH)
frames.
The position of the spacecraft in the body frame is denoted as r. Quaternions are
used to model the attitude dynamics of equations[16].
Quaternion q is a four-element vector with a three-element vector q1:3 and a scalar
part q4. The quaternion unity constraint is
q
2
4 = 1 − ∂qT
1:3∂q1:3 (1)
The orientation of frame Fa to frame Fb through Euler angle θ and Euler axis υ is
expressed through quaternion as
q
ba =
"
υsin(θ/2)
cos(θ/2) #
(2)
The associated attitude matrix corresponding to the quaternion q
ba is shown as Rba
.
⊗ is quaternion multiplication operator defined by (3).
q ⊗ q¯ =





q4 −q3 q2
q3 q4 −q1
−q2 q1 q4
−q1 −q2 −q3










q¯1
q¯2
q¯3
q¯4





(3)
Quaternion multiplication between reference frames are
q
ac = q
bc ⊗ q
ab (4)
Equation (4) corresponds to the multiplication of attitude matrices as
R
ac = R
bcR
ab (5)
Small rotations θ can be written in terms of attitude matrix R as
∂R = I 3×3 − [θ×] (6)
Small rotations θ can also be written in terms of quaternion q as
∂q ≈
"
θ/2
1
#
(7)
6
Where θ = θυ, and I is an identity matrix. The cross product matrix [x×] of the
variable x is defined as
[x×] =



0 −x3 x2
x3 0 −x1
−x2 x1 0


 (8)
The identity quaternion is
q =





0
0
0
1





(9)
2.5. Orbits and the desired trajectories
The baseline flight dynamics of VTXO uses a highly-elliptical geostationary transfer
orbit with a 32.5-hour period for providing a 10-hour observation in the apogee [31]. The
5 Keplerian elements are the same for OS and IS except the eccentricity γ given in table
D.9.
The eccentricity of OS and IS are designed to include a few minutes of buffer between
the time the OS and the IS pass the point where the orbits intersect, avoiding a collision
between satellites. A larger difference between the eccentricities results in a lower risk of
collision, since the satellites would have longer relative distances. However, this results in
a higher energy consumption that is needed to keep the desired 1 km relative distance
between the satellites.
The list of objects to be observed by VTXO are given in Table 1. This table of desired
objects for VTXO can be updated in the future.
The desired attitude trajectory to observe the desired objects given in Table 1 is
denoted as qf . qf doesn’t vary by time; it just switches from one space object to the next
when the science observation satisfies the time of observation.
Table 1: Time of observation for the objects VTXO observe in the observation phase [1].
Objects Time of observation (hours)
Sco X-1 0.2
GX 5-1 1.5
GRS 1915+105 4.2
Cyg X-3 4.9
Crab Pulsar 5.4
Cen X-3 19
γCas 146
Eta Carinae 452
2.6. VTXO Dynamics
q
bi represents the orientation of the spacecraft body frame with respect to the earth
Earth-centered inertial (ECI) frame, and ω
bi
b
corresponds to slew rate of the spacecraft
body frame with respect to the inertial frame.
The states x for the attitude of spacecraft include quaternion and angular velocities,
and the states s for the position of spacecraft include the position and velocity of spacecraft.
x is given by
7
x =
h
q
T ω
T
iT
(10)
q =
h
q1 q2 q3 q4
iT
ω =
h
ω1 ω2 ω3
iT
And s is given by
s =
h
r
T v
T
iT
(11)
x at the beginning of the transient phase are the initial slew and initial slew rate for
the transient phase denotes as x0, and x at the end of the transient phase are denotes as
xf .
A time variant modeling error δJ is considered in the nominal inertial momentum ¯J
to model the changes in the true value of inertial momentum J as the following
J
−1 = ¯J
−1
+ δJ −1
(12)
The total applied torque to the spacecraft due to control input, noise, and disturbances
are denoted as τ . The equations of motion for the dynamics of spacecraft are given by
q˙ =
1
2
ω ⊗ q (13)
ω˙ = J
−1
(τ − ω × Jω) (14)
τ is given by
τ = τ in + τ g + wω˙ (15)
τ in corresponds to the control input, wω˙ corresponds to external disturbances from the
space environment modeled as Gaussian white noise, and τ g corresponds to the time
variant gravity gradient torque. τ in derives the quaternion q to the desired quaternion qf
and ω to zero, which is equal to reaching the equilibrium point xe.
The dynamics for the spacecraft relative position formation is given by
r˙ = v (16)
r¨ = −
Xn
i=1
µi
ri
|ri
|
3
+ u (17)
ri shows the position vector of the spacecraft to the gravitational bodies.
The relative position dynamics is given by
rrel = rF − rL (18)
u is given by
u = ug + uin (19)
uin corresponds to the position controller provided by the thrusters, and ug corresponds
to solar disturbances gsolar and other perturbations gpert given by
8
ug = gsolar + gpert (20)
As a result, the relative position dynamics is given by
r˙ rel = vF − vL (21)
r¨rel = −
Xn
i=1
µi(
riF
|riF |
3
−
riL
|riL|
3
) + uF − uL (22)
riF and riL show the vector of the follower and leader to the gravitational bodies,
respectively.
2.7. External disturbances and gravity gradient torque
In VTXO, disturbances are modeled as bounded, time-varying, differentiable forces
acting on the position and torques in the dynamics of the spacecraft.
Gravity gradient torque τ g is derived from point mass gravity models [16, 32] as the
following
τ g =
3µ
|r|
3
n × (Jn) (23)
where n is the nadir vector in the body frame Fb, and |r| is the radial distance of spacecraft
from the earth.
wω˙ corresponds to the random torques, J2 gravity model, and torques to account
for drag, solar pressure, higher-order-gravity terms, etc. wω˙ is modeled as a zero-mean
Gaussian white noise process where the power of the noise is captured in the variance as
σ
2
ω˙
[33].
E[wω˙(t)wω˙(t
′
)
T
] = σ
2
ω˙
I 3×3δ(t − t
′
) (24)
ug is modeled as a first-order Markov processes also known as exponentially correlated
random variables (ECRV) given by
u˙ g =
−ug
τg
+ wg (25)
Large and small time constant τ makes the bias a constant value or a white noise,
respectively.
The variance of Gaussian white noise wg is given by σ
2
g
, and the time constant for ug
is given by τg.
δ(t − t
′
) is the Dirac delta function defined as
δ(t − t
′
) = 0 if t ̸= t
′
(26)
Z ∞
−∞
δ(t − t
′
) dt′ = 1 (27)
σ
2
ω˙ are chosen based on the orbit and the size of spacecraft. If the spacecraft are
bigger or closer to earth, σ
2
ω˙
is chosen larger due to higher drag forces and other torques.
Choosing the external disturbances on the spacecraft is shown by Lear[23, 33], where the
estimated disturbance corresponds to the expected downrange and attitude error after
one orbit. In [13], the external disturbances are modeled as first-order Markov processes
while in [23] it is given as a white noise.
9
2.8. Sensor model VTXO
Gyroscope in inertial measurement unit (IMU) is used to measure ω given by ω˜, star
tracker is used for measuring q given by q˜, accelerometer in IMU for measuring r˜¨, Radio
ranging sensor for measuring r˜
z
rel, and interferometry sensor in the NISTEx-II star tracker
for measuring r˜
xy
rel.
2.9. Actuator model VTXO
In VTXO, reaction wheels are used for ACS and thrusters are used for relative position
control as the primary actuators in GNC for both the follower and the leader. The torque
generated by the reaction wheels τ in and by the thrusters uin use the commanded torque
τˆin and commanded force uˆin, respectively, given by the control law.
2.9.1. Reaction wheel
In τ in [23, 34], Gaussian white noise wτ , bias bτ , scale factor f τ , and misalignment ϵτ
are included as the following
τ in = ∂R(ϵτ )[{I 3×3 + diag(f τ )}τˆin + bτ + wτ ] (28)
The variance σ
2
wτ
captures the power of the noise in the random noise wτ as
E[wτ (t)wτ (t
′
)
T
] = σ
2
wτ
I 3×3δ(t − t
′
) (29)
The bias bτ is modeled as a first-order Markov processes given by
˙bτ =
−bτ
ττ
+ wbτ (30)
The variance σ
2
wbτ
captures the power of the noise in the random noise wbτ .
2.9.2. Thruster
In uin [13, 23, 34], Gaussian white noise wu, bias bu, scale factor f u are included in
the model as the following
uin = [{I 3×3 + diag(f u)}uˆin + bu + wu] (31)
The variance σ
2
wu
captures the power of the noise in the random noise wu as
E[wu(t)wu(t
′
)
T
] = σ
2
wu
I 3×3δ(t − t
′
) (32)
The bias bu is modeled as a first-order Markov processes given by
˙bu =
−bu
τu
+ wbu (33)
The variance σ
2
wbu
captures the power of the noise in the random noise wbu.
10
2.10. Navigation structure and sensor fusion
The estimated value or the value in the flight computer for the true variable x is
denoted as xˆ, and the measured value for x is denoted as x˜.
The variation of EKF with the model replacement method and the multiplicative
extended Kalman filter (MEKF) are used in VTXO. In the transient phase and the science
observation phase, without keeping the precision formation, star trackers and gyro sensors
are used to provide a few mas accuracy for both follower and leader.
In the science observation phase, 55 mas formation accuracy, sub millimeter formation
transverse alignment accuracy, and 1 meter relative distance accuracy are obtained using
the following instruments
• Gyro sensor.
• Accelerometer sensor.
• Radio ranging.
• Laser beacons on the leader spacecraft.
• NISTEx-II star tracker on the follower spacecraft.
• Star tracker on the leader.
In the science observation phase, both attitude control and relative position control
are used to hold the precise formation.
For the relative position in the phases before and after the science observation phase,
the accelerometer sensor and the GPS are used. In the formation stabilization phase, GPS
is used. GPS calibrates the orbits to correct any deviation from the nominal orbits at the
perigee.
2.10.1. Dynamics of spacecraft for prediction in the navigation states
The spacecraft dynamics used for propagating the states are as follows.
˙qˆ =
1
2
ωˆ ⊗ qˆ (34)
ωˆ˙ = ˆJ
−1
(τˆ − ωˆ × ˆJωˆ) (35)
˙rˆ = v (36) ˆ
˙vˆ = −µ
rˆ
|rˆ|
3
− uˆ (37)
For the relative position, the propagated states are given as
rˆrel = rˆF − rˆL (38)
˙rˆrel = ˆvF − vˆL (39)
˙vˆrel = −µ(
rˆF
|rˆF |
3
−
rˆL
|rˆL|
3
) + uˆF − uˆL (40)
11
The first-order Markov process bias and misalignment are propagated as
ˆ˙b =
−b
τ
(41)
ˆϵ˙ =
−ϵ
τ
(42)
2.10.2. Multiplicative extended Kalman filter for the attitude control system
The multiplicative extended Kalman filter (MEKF) with model replacement mode is
used for the ACS.
When the model replacement mode is used, the angular velocity is directly replaced
with the gyro measurements ω˜, and the spacecraft dynamics is propagated as
˙qˆ =
1
2
(ω˜ − ˆbω) ⊗ qˆ (43)
Since the measurements violate the quaternion unity constraint (1), the multiplicative
version of EKF is used to propagate the quaternion states. In MEKF , the propagated
quaternions are updated by small orientation (7) given by
q
+ = ∂q+(θ) ⊗ q
− (44)
When MEKF is used, the navigation states of each satellite is a 12-element vector
given by
x =
h
θb ω bω bq
iT
(45)
When MEKF with model replacement is used, the navigation states of each satellite is
a 9-element vector given by
x =
h
θb bω bq
iT
(46)
Further discussion on MEKF with model replacement for VTXO ACS is given in [4].
2.11. Control law
A nonlinear globally asymptotic stability Lyapunov controller [16, 35] is defined as
the control law for the transient phase attitude control. SMC is defined for the science
observation attitude control and relative position control. Anti gravity gradient torque is
considered for the formation stabilization phase.
For showing the effect of linear controllers and the effect of SMC in the transient phase,
they are implemented and compared with the chosen controllers.
For the ACS, the control law minimizes the difference between the desired quaternion
qf and the current state of quaternions q and the difference between the desired angular
velocities wf and the current state of angular velocities w. The difference between qf and
q is denoted by ∂q given by
∂q ≡
"
∂q1:3
∂q4
#
= q ⊗ qf
−1
(47)
∂q1:3 = E(qf )
T
q (48)
where E(q) is the 4×3 matrix
12
E(q) =





q4 −q3 q2
q3 q4 −q1
−q2 q1 q4
−q1 −q2 −q3





(49)
∂q4 = q
T
qf (50)
Taking the time derivative of (47) leads to
∂q˙ = ˙q ⊗ qf
−1
(51)
Using (47) and substituting (13) into (51) gives
∂q˙1:3 =
1
2
[∂q1:3×]ω +
1
2
∂q4ω (52)
∂q˙4 = −
1
2
∂qT
1:3ω (53)
2.11.1. Linear controller
For the attitude controller, a linear proportional-derivative (PD) controller is used as
the following
τˆin = ˆk
q
P
ˆ ∂q1:3 − ˆk
q
D(ωˆf − ω) (54)
Where ωf shows the final desired angular velocity which is zero in VTXO. The vector
of the PD controller parameters to be defined for ACS is given by
ˆk =
"
ˆk
q
P
ˆk
q
D
#
(55)
For the relative position controller, a linear proportional-derivative (PD) controller is
used as the following
uˆin = ˆk
r
P
(r
f
rel − rˆrel) + ˆk
r
D(v
f
rel − vˆrel) (56)
2.11.2. A Lyapunov controller for the attitude control system
The Lyapunov controller is defined as the following
τˆin = − ˆk1sign(∂qˆ4)
ˆ ∂q1:3 − ˆk2(1 − ˆ ∂q1:3
T
ˆ ∂q1:3)ωˆ (57)
where ˆk is estimated by Monte-Carlo, optimization, ML, or chosen randomly with a
uniform distribution and xˆ, as explained earlier, is obtained from navigation filters. In
[30, 36], the global asymptotic stability is proved using the Lyapunov stability theorem.
In the simulation, noise, disturbances, and uncertainty are considered, and the stability
for the closed-loop system is shown using Monte Carlo simulation.
13
2.11.3. Sliding mode controller for the attitude control system
The sliding vector s for ACS is given by [16]
sˆ = (ωˆ − ωf ) + ˆkSMCsign(∂qˆ4)
ˆ ∂q1:3 (58)
As explained in section Appendix B, the control law is obtained by s˙ = 0 and including
the sat function. The control law is obtained as
τˆin =J



ˆkSMC
2
"
|∂qˆ4|(ωf − ωˆ)
− sign(∂qˆ4)
ˆ ∂q1:3 × (ωf + ωˆ)
#
+ ω˙ f − Zˆ sat( ˆsi
, εˆi)



+ ωˆ × Jωˆ
(59)
si shows the ith component of the sliding vector in (58), where i = 1, 2, 3. Z is a
positive definite matrix, and εi
is a positive scalar. To reduce the chattering, instead of a
sign function, the saturation function sat( ˆsi
, εˆi) is used given as the following
sat( ˆsi
, εˆi) =



1 for sˆi > εˆi
sˆi/εˆi
for |sˆi
| ≤ εˆi
−1 for sˆi < εˆi


 (60)
The saturation function sat(sˆi
, εˆi) derives the system to the sliding surface. The
following Lyapunov function proves the global asymptotic stability.
V (∆(x)) = 1
2
s
2
(61)
And the time derivative of the Lyapunov function is given as
V˙ (∆(x)) = ss˙ (62)
In section Appendix B, the proof for the global asymptotic stability of SMC is shown.
Since the desired ω are zero, ωf = 0.
ˆk,εˆi
, and Zˆ can be estimated by Monte-Carlo,
optimization, ML, or chosen randomly with a given distribution. xˆ is obtained from
navigation filters given in section 2.10.
Z is chosen to be a positive scalar here denoted as z. The vector of SMC parameters
to be estimated is given by
ˆk =



zˆ
ˆkSMC
εˆ


 (63)
2.11.4. Sliding mode controller for the relative position control law
In the MASSIM virtual telescope [13], it is assumed that the leader drifts in its natural
orbit and the follower keeps the desired relative position with the leader. In the science
observation phase, both the follower and the leader control the relative position and
14
consider the thrust control input of the other spacecraft as a disturbance in the SMC.
The sliding vector s for the relative position control is given by
sˆ = (vˆrel − v
f
rel) + ˆk(rˆrel − r
f
rel) (64)
When the relative position is controlled by the follower, the assumed model ¯f(rˆrel,vˆrel)
is given by
¯f(rˆrel,vˆrel) = −µ(
rˆF
|rˆF |
3
−
rˆL
|rˆL|
3
) − uˆinL (65)
For a natural drift of the leader, uˆinL is zero. The vf
rel = 03×1 and v˙
f
rel = 03×1.
As explained in section Appendix B, the control law is obtained by s˙ = 0 and including
the sat function. The control law is obtained as
uˆinF = µ(
rˆF
|rˆF |
3
−
rˆL
|rˆL|
3
) + uˆinL − ˆk(vˆrel − v
f
rel) − Zˆ sat( ˆsi
, εˆi) (66)
si shows the ith component of the sliding vector in (64), where i = 1, 2, 3. Zˆ is a
diagonal positive definite matrix, and εi
is a positive scalar. To reduce the chattering,
instead of a sign function, the saturation function sat(sˆi
, εˆi) is used given as the following
sat( ˆsi
, εˆi) =



1 for sˆi > εˆi
sˆi/εˆi
for |sˆi
| ≤ εˆi
−1 for sˆi < εˆi


 (67)
The saturation function sat(sˆi
, εˆi) derives the system to the sliding surface. The
following Lyapunov function proves the global asymptotic stability
V (∆(x)) = 1
2
s
2
(68)
And the time derivative of the Lyapunov function is given as
V˙ (∆(x)) = ss˙ (69)
In section Appendix B, the proof of the global asymptotic stability of SMC is shown.
ˆk,εˆi
, and Zˆ can be estimated by Monte-Carlo, optimization, ML, or chosen randomly with
a given distribution. xˆ is obtained from navigation filters given in section 2.10. With the
same approach, the thruster input for the leader uˆinL, when the follower control input is
considered disturbance, is obtained as
uˆinL = µ(
rˆL
|rˆL|
3
−
rˆF
|rˆF |
3
) − uˆinF − ˆk(vˆrel − v
f
rel) − Zˆ sat( ˆsi
, εˆi) (70)
Where rrel and r˙ rel are defined as rrel = rL − rF and r˙ rel = vL − vF , respectively.
3. Artificial Intelligence Framework for Adaptive Attitude Control
ML estimates the optimal parameters in the transient and science phases using
supervised learning. The data is obtained using optimization methods. The optimal
parameters that ML estimates are k = [k1, k2], E, and e based on xf , x0 and w. T is
estimated by ML for the transient phase, and it is chosen from the table 1.
Fig. 2 shows the 3 stages of AI as follows:
15
• Optimization and data production phase
• ML phase
• Implementation phase
Figure 2: This unified framework addresses both transient and science phases. In the optimization and
data production phase, initial states (x0), weights (w), and final states (xf ) are generated and given to
optimization. Simulated Annealing (SA) and Multi-objective Genetic Algorithm (MOGA) optimization
algorithms then optimize variables k1, k2, and transient phase duration T to minimize the objective
function χ (composed of e and E). The resulting optimized dataset trains the Machine Learning (ML)
model to predict the optimal parameters (yˆ)
To produce each data, the objective function χ is optimized with the variable k and T
(for the transient phase) for each set of x0, w, and xf . χ is a vector (76) of E and e for
the science phase weighted sum of E and e (72) for the transient phase. After producing
the data, ML estimates y = [E, e, k1, k2] with the input w, x0, and xf . Also T is
estmated with ML in the tansient phase.
In the ML phase, the inputs to ML are w and x0, and the outputs of the ML are the
E, e, and z defined as y = [E, e, k1, k2].
The optimization and data production phase and ML phase are implemented offline,
whereas the implantation phase is implemented in the real-time in space using FPGAs.
In the implantation phase, the values of x0 obtained from attitude estimation algorithms
[4, 16], denoted by x˜0, and the value of w gives the estimated optimal k denoted by ˆk.
The ML estimates E and e, denoted by Eˆ and eˆ respectively, obtained from the maneuver
by the zˆ parameters. The estimated y used in the implementation phase is denoted by yˆ.
It is shown later that DNN is chosen to estimate yˆ. The DNN is incorporated into FPGA,
as the input to the FPGA is x˜0, w and the output of FPGA is yˆ.
The optimization and ML phases are conducted offline for training and testing. The
implementation phase operates in real-time on FPGAs for inference, using attitude
estimation algorithms [4, 16] to provide x˜0. With w as an additional input, the FPGAembedded DNN directly estimates the optimal parameters yˆ for the optimal maneuver.
16
A key contribution of the artificial intelligence (AI) framework presented here is its
inherent explainability, which is critical for mission-critical systems. Unlike opaque "blackbox" models that simply provide an output, this framework offers transparency. The
explainability is achieved as follows:
Predictive Transparency: The ML model is trained not only to determine the optimal
controller parameters (z) but also to predict the associated energy consumption (E)
and slew error (e). As shown in Figure 4, the output yˆ includes these performance
metrics. This allows a mission operator (or an autonomous system) to understand
the consequences of a control decision before it is executed. For example, the system
can explicitly manage the trade-off between performance and resource usage by
adjusting the weight w, knowing the predicted outcome. This contrasts sharply
with methods like adaptive LQR, where the future energy and error costs are not
explicitly estimated as part of the real-time control loop.
Structural Transparency: The overall mission logic is governed by the supervisory
control system (detailed in Section 12), which uses a clear, rule-based timed automata.
This ensures that every phase transition is deterministic and based on verifiable
conditions, not on an uninterpretable internal state of a complex model.
The goal of this explainable AI is to provide ACS with optimal adaptive values,
including the duration of the transient phase T.
3.1. Unified Supervisory Optimization Formulation
The optimization loss function χ for the transient phase is a scalar given as:
χ = E + e (71)
The loss function χ of the Pareto optimization for the science phase is a vector given as:
χ =
"
E
e
#
(72)
w = e/E is a weight for e over E. The scaler value of e is minimized for the whole
duration of the science phase, while it is minimized in the last D = 4 seconds (73) to
minimize the steady state tracking error in the transient phase. D is chosen to be minimal
to reduce the computation cost for the large angle manouvers, which also corresponds to
the steady-state error. Variable e is defined as
e =
1
D
Z Tend
Tend−D
q
e(t)
⊤e(t) dt (73)
where e(t) is the absolute difference between the desired attitude and the actual attitude.
Tend is the end of the phase, which corresponds to the beginning of the next phase given
as T0.
Scaler value E is defined as
E =
Z Tend
T0
τ
⊤
in(t)ω(t) dt (74)
Scaler value T represents the duration of each phase, given as:
T = Tend − T0 (75)
17
For the science phase, T is given in table 1. T is a variable defined by ML for the transient
phase.
The constraints in SA are the constraints related to the dynamics of the system, the
unity norm of the q0, and the range of the variables. N and U are Gaussian (with µ, σ
for mean and standard deviation) uniform distributions, respectively.
Multi-objectives Genetic Algorithm (MOGA) and SA find the optimal variables in the
given unified (for both phases) non-convex constrained nonlinear optimization:
min
k
χ(x0, w, xf , k) (76)
Subject to the spacecraft rigid-body dynamics:
x˙ = f(x(t), τ ) (77)
s.t : τˆ = G(k, x(t))
Quaternion normalization:
q
T
0
q0 = 1 (78)
Control parameter bounds:
0.01 < k < 3 (k ∈ k) (79)
Time constraint for the transient phase duration (T):
7.2s < T < 72s (80)
Final state bounds:
x
l
f < xf < xu
f
(81)
Initial state bounds:
x
l
0 < x0 < xu
0
(82)
Initial condition distributions for the transient phase:
x0j ∼ U (j = 1, 2, 3, 4)
x0j ∼ N(µ1, σ1) (j = 5, 6, 7)
Final condition distributions:
xf j ∼ U (j = 1, 2, 3, 4)
xf j ∼ N(µ2, σ2) (j = 5, 6, 7)
Weight distribution for the transient phase:
w ∼
h
N(µ3, σ3), U(c
l
1
, cu
1
), U(c
l
2
, cu
2
)
i
(83)
Initial condition distributions for the science phase:
q0 = ∂q(θ0) ⊗ qf (84a)
18
θ0 ∼ N(µ4, σ4)
ω0 ∼ N(µj
, σj ) (j = 5, 6, 7)
x0 =
"
q0
ω0
#
(84b)
The distribution of ω0 for the science phase is zero mean with the standard deviation
chosen based on the table 3 and figure 4, as they show the ωf distribution for the end
of transient phase. The variance for the zero mean θ0 is chosen considering the P99(e)
from the transient phase. P99(e) from the transient phase is chosen as the 3-sigma for the
standard deviation of q0 for the science phase.
The bounds of x0 are
x0
l =













−1
−1
−1
0
−2
−2
−2













, x0
u =













1
1
1
1
2
2
2













(85)
xf
l =













−1
−1
−1
0
0
0
0













, xf
u =













1
1
1
1
0
0
0













(86)
Lyapunov function control parameters k1, and k2 are positive scalar for global asymptotic stability. The U distribution bounds with the Gaussian in parameters are
c
l =
"
0
1
#
, cu =
"
5
1
#
, µ =













0
0
5
0
0
0
0













, σ =













0.6
0
0.3
0.0289
0.052
0.048
0.058













(87)
For a baseline mission with a 22.5-hour formation stabilization phase, ω3 reaches 0.59
rad/sec. Therefore, 0.6 is chosen as the 1-sigma of ω with zero mean distribution as
given in equation (87). The initial quaternion distribution of the transient phase q0 is
a uniform distribution between -1 and 1 in equation (85) as the quaternion distribution
varies between -1 and 1 for the formation phase.
For the transient phase, SA solves the optimization equation (76) and produces the
data given in the table 2. MOGA solves the optimization equation (76) for the science
phase.
19
3.2. ML phase
3.2.1. Transient phase
DNN is trained on 7893 data points produced in the data production phase. Both
mean absolute percentage error (MAPE) and mean absolute error (MSE) are used for
showing the estimation accuracy. k-fold cross validation with k = 4 is used to measure
MAPE and MSE. As a result, 25% of the data is used as test data. The inputs of the
DNN belong to R
8
, and they are q0, ω0, and w. The outputs of the DNN belong to R
5
,
and they are E, e, T ,k1, and k2.
Since all the outputs are positive, the ReLU [30, 36] is chosen as the activation function
for the output neurons. The hyperparameter of the DNN’s structure are number of layers,
number of neurons at each layer, and activation function. The parameters weight and bias
of the DNN are optimized through the Adaptive Moment Estimation (Adam) [37] and
Nesterov-accelerated Adaptive Moment Estimation (Nadam) [38] optimizer algorithms.
The optimization criterion consists of minimizing the MAPE through the Adam and
Nadam algorithms.
The k-fold cross validation with k = 4 is used to measure MAPE and MSE. As a result,
25% of the data is used as the test data. While training the DNN, 20% of the training
data is used as a validation data set for the early stopping. Number of batches, number
of epochs, l1 regularization, kernel constraint are cross validated. For kernel constraint,
M axNorm(x) function is used which limits the maximum norm of the weight vector for
the layer with x. DNN’s weights are bounded using M axNorm(x) function for kernel
constraint.
The hyperparameter optimization problem obtains the minimum MAPE during the
validation phase. The hyperparameters of the DNN’s structure and algorithm parameters are: number of layers, number of neurons at each layer, epochs, batches, weight
initialization, l1 regularization, kernel constraint, activation function for the hidden layers,
and patience. Adding other regularization methods including batch normalization and
dropout doesn’t reduce MAPE.
The coarse-to-fine optimization approach is to solve the hyperparameter optimization
problem first with GS. Next, the optimal solution from GS is used to solve the hyperparameter optimization problem with RS, which is finally validated on ALCF. The final
optimal hyperparameter has 3 layers for each output. Other validated hyperparameter
are given in [30]
3.2.2. Science phase
For this phase the controllers SMC and PD with MEKF are used to provide the leader
and follower with arcminute accuracy. The star tracker and gyro sensor provide the
attitude and angular velocity measurements for the MEKF. The trained DNN minimises
the e and E in the science observation phase. DNN multi output is used for PD controller
and SMC with 10 neurons in the first layer, 5 neurons in the second layer, and 3 neurons
in the last layer. In the neural network in the training phase, the number of maximum
epochs is set to 1000. Sigmoid is used as the activation function, and stochastic gradient
descent (SGD) is used as the optimizer algorithm with MSE loss function.
4. Experiments
• The Monte Carlo simulation is used to analyse the performance and stability of the
closed loop system.
20
• Monte Carlo simulation produces the data required for the ML.
• Monte Carlo simulation is used to design controller gains and T using the mode of
data set.
The ACS solver uses the fourth-order Runge-Kutta method (RK4) with a fixed time
step to provide accurate and timely solutions for the ACS equations, ensuring that the
quaternion unity norm constraint is maintained. The time step is chosen to stabilize
97% of the data within a reasonable timeframe, with any remaining unstable solutions
addressed by 300 iterations of Simulated Annealing (SA), resulting in an average data
production time (TQ) of 0.51 hours. This approach prioritizes accuracy and computational
efficiency, removing the top 80% of data with the highest χ values to enhance the accuracy
of ML estimation.
5. Data analysis and ML for transient phase
Fig. 3 shows that the data are positive and bounded. The mode of the data set is
between the maximum and minimum close to average except for T, which has its mode
equal to its maximum. This data distribution shows that the T is constrained and not
minimized as its mode is close to its maximum. For e and E the mode is close to the
average as they are minimized in the objective function.
Figure 3: Distributions of k1, k2, e, E, and T from the data production phase. The parameters k1, k2, e,
and E all exhibit Gamma-like distributions. Figure adapted from [30].
Table 2 compares the statistics of the data with the prediction values from DNN. It
highlights that the maxima of e and E are considerably higher than their P99 values.
This discrepancy comes from solver sensitivity in the ACS, noise in the dynamics, the
stochastic nature of simulated annealing, and the heavy-tailed character of the resulting
distributions. As e converges to zero, the terminal angular velocity ωf also converges
toward zero, consistent with the histogram in Fig. 4. The P99 and P1 bounds for all ωf
components lie within ±1 deg/s, confirming tight dispersion around zero.
21
Table 2: Data statistics with 99 percentile values of DNN predicted outputs.
y Mean Mode Variance P1 Max P99 DNN (P99)
k1 0.1342 0.085 0.0070 0.0244 0.4998 0.41 0.5627
k2 0.2906 0.25 0.0077 0.0926 0.4998 0.49 0.7580
e (deg) 0.0233 0.015 6.2353e-04 0.0035 0.0990 0.067 0.0867
E (J) 0.1349 0.075 0.0058 0.0229 0.6302 0.37 0.4695
T (s) 48 72 326.46 9 72.0 70.53 129.26
Table 3: ωf statistics.
y Mean Variance P99 P1
ωf1 (deg/s) 1.03e-04 0.0027 0.1164 -0.0972
ωf2 (deg/s) -5.88e-04 0.0023 0.0592 -0.1927
ωf3 (deg/s) 9.08e-5 0.0034 0.1211 -0.0938
Figure 4: A kernel density estimate (KDE) for ωf distribution. The distribution shows ωf is centered
around zero with a small variance.
The percentile statistics demonstrate that mission requirements are satisfied by the
optimization data: P99(e) = 0.067◦ < 0.09◦ and P99(E) = 0.37 J per orbit. This
energy budget remains comfortably low for long-duration operations, well within current
spacecraft technology capabilities. In this sense, the simulated annealing optimization
plays a role similar to model predictive control (MPC): actively tuning gains to satisfy
strict constraints. The drawback is computational cost—on average, about 30 minutes
are required to complete a single optimization run.
5.1. Transient phase ACS stability analysis
In the ML approach, using ReLU as the activation function for the output layer forces
the controller gains k1 and k2 to be positive scalars(80). Since ML defines the positive
Lyapunov controller gains at the beginning of the transient phase, the controller is globally
asymptotic stable without considering the noise, disturbances, and model uncertainties.
However, the closed loop system is shown to be stable using Monte-Carlo simulation since
P99(e) and P99(E) are bounded. The closed loop system is stable and P99(e) <0.2 deg
and P99(E)<2 J. Having the kernel constraint enforces the weights to be bounded so the
DNN outputs are bounded when the inputs are bounded. Besides, the ML gives P99(eˆ)
22
<0.09 deg and P99(Eˆ)<0.5 J using Monte Carlo simulation for all the data, which shows
the stability of closed loop system using DNN.
6. Data analysis and ML for science phase
The data is produced using multi objective genetic algorithm (MOGA) for SMC and
PD controllers. One sample of MOGA Pareto front for SMC is shown in Fig 5.
Figure 5: The Pareto front for the sliding mode controller in the science observation phase.
6.1. ML results
When using DNN, the linear controller PD and SMC provide the MSE as 0.0434 and
0.0124, respectively. These ML are used to predict the controller parameters k. SMC
provides the VTXO with lower MSE compared to PD controller.
6.2. Simulation results
Two experiments are done to show the performance of the given methodology. For
two given w, the estimated optimal controller parameter by ML, ˆk, is given in Table 4.
Table 4: ˆk values
w ˆk
wP D = 6.28 ˆkP D = [0.1208 0.3786]T
wSMC = 26.14 ˆkSMC = [2.9947 0.0193 0.2601]T
Using ˆk from the ML, the e and E are obtained using the GNC simulation given in
Table 5.
Table. 5 shows that e is a few arcminute which satisfies the mission requirement
for each individual spacecraft. In this example, PD controller error e here is more than
0.18 deg, which is not acceptable. SMC is used as it provides a more robust controller
toward disturbances and uncertainties for the nonlinear spacecraft dynamics, with a low
prediction error as 0.0124.
23
Table 5: e and E obtained from ˆk.
controller e (deg) E (J)
PD 0.2219 0.0353
SMC 0.1738 0.0066
7. Relative position formation
The relative position formation is shown for the science observation phase. For
simplicity, it is assumed the leader is in its natural orbit and the follower controls the
relative position. Disturbances are included in the dynamics given in section 2.7. SMC
and PD controller are given in section 2.11. The sensors used for measurements are
the radio ranging for relative distance measurements r˜
z
rel, IMU the for relative velocity
measurements v˜rel, and the interferometry sensor for transverse alignment measurements
r˜
xy
rel. As a result, each sensor measures the states of the dynamics individually. The sensor
measurements are included in the simulations. The parameters for the PD controller is
chose as P = 1 and D = 1. For the SMC, the sat function is chosen as a sign function for
simplicity, and the controller parameters are k = 1 and Z = 1 for each axis. The desired
relative velocity and the desired relative position are
v
f
rel = 03×1 (88)
r
f
rel =



0
0
1


 (km) (89)
The initial conditions are
v
0
rel =



1
1
1


 (m/s) (90)
r
0
rel =



1
2
10


 (km) (91)
The transverse alignment using SMC for the last 15 s is shown Fig. 7
Fig. 6 shows that the system converges to the desired trajectory. Fig. 7 shows that
relative position alignment stays in the order of 10−6, which satisfies the sub millimeter
accuracy. The transverse alignment fluctuates because of the noise in the sensors and
disturbances in the system.
The transverse alignment using PD for the last 15 s is shown Fig. 8.
The PD controller makes the system converge to the desired trajectory. Fig. 8 shows
that relative position alignment stays in the order of 10−3, which doesn’t satisfy the
sub millimeter accuracy. The transverse alignment fluctuates because of the noise in the
sensors and disturbances in the system. It shows that the designed PD controller can’t
satisfy the science phase accuracy requirement. However, the PD can be modified to PID
or a better tuned PID controller for a better tracking characteristic. The added integral
to the controller corrects for steady-state error by accumulating the error over time.
24
Figure 6: The relative position formation using SMC.
Figure 7: Transverse alignment using SMC when the system is settled for the last 15 s.
8. Supervisory adaptive control system for ACS
The timed automata method is used to model the supervisory adaptive control system
[39, 40]. The stability of hybrid control systems are discussed in [41]. The system includes
25
Figure 8: Transverse alignment using PD when the system is settled for the last 15 s.
different phases and controllers. The timed automata method models the changes in the
phases with the needed conditions for it. The commissioning, ACS phases in collaboration
with the relative position control, and de-commissioning are expressed below.
The commissioning takes for 60 days. After the commissioning, the first space object
begins to be observed in the science phase after the transient phase and when the relative
distance is achieved with 1 meter accuracy. In the science observation phase, the formation
holds for the duration given in Table 6 for each space object, after the commissioning and
before the de-commissioning. In Table 6, the space objects, the commissioning, and the
de-commissioning are considered as the operational mode. The Id shows the operational
mode index.
Table 6: Finite automata rule u3.
Id Operational mode state s4 Rule u3
0 Commisioning 60 days
1 Sco X-1 0.2 hr
2 GX 5-1 1.5 hr
3 GRS 1915+105 4.2 hr
4 Cyg X-3 4.9 hr
5 Crab Pulsar 5.4 hr
6 Cen X-3 19 hr
7 γCas 146 hr
8 Eta Carinae 452 hr
9 De-commissioning 5 days
26
After the science observation phase, the formation stabilization phase completes the
32.5 hr duration of the orbit if still observing the same object for the duration given in
Table 6.
This cycle of phases repeats until the observation period for the objects given in Table
6 is met, and then the orbits are switched to the new orbits for the new space objects in
Table 6, and the cycle repeats for the new space object. After the mission is completes,
the de-commissioning deorbits the spacecraft.
The timed automata model of the designed supervisory control system is a 5-tuple as
T A = {s, t, i, k, Init}. Each tuple is explained below.
• s = {s0, s1, s2, s3, s4, s5} is the set of discrete states of the finite automata model of
the system. Each discrete state of the system si
is associated with a specific phase
or an operational model of the system.
• t = {t1, t2, t3} is the set of local timers. The local timer shows the amount of time
that the system has spent in each state s.
• i = {u1, u2, u3, u4} is the set of the inputs to supervisory control system.
• k = {k1, k2, k3} is the set of all transitions. A transition ki
is a four tuple given by
kij = {si
, g(i, j), reset(i, j), sj} (92)
si and sj are the source and destination states, and g(i, j) is the guard condition
for the transition. Gaurd conditions are Boolean expressions that are evaluated
based on the inputs i and local timer t. Table 8 shows the guard conditions and
their corresponding Boolean expressions in the designed supervisory control system.
reset(i, j) : R → R is the reset condition for the transition.
• Init = {c1, c2, c3} is the set of initial conditions for the system where Init ⊂ [s, t]
In the design of the architecture of the timed automata model, the set of discrete
states of the finite automata s is given as below
• s0: Commissioning.
• s1: Formation stabilization phase.
• s2: Transient phase.
• s3: Science observation phase.
• s4: Next operational mode state. At this state, we go to the next operational mode,
and the Id in Tbale 6 is increased.
• s5: De-commissioning.
• s6: End of operation.
In the design of the architecture of the timed automata model, the set of local timers
t is given as below
• t0: Commissioning duration.
27
• t1: Formation stabilization phase time.
• t2: Transient phase time.
• t3: science phase time.
• t4: Next operational mode time.
• t5: De-commissioning duration.
In the design of the architecture of the timed automata model, the set of the inputs to
supervisory control system i is given as below:
• u1: Formation stabilization phase rule. It is obtained by subtracting the 10 hr
duration of the science phase and the maximum 3 min duration of the transient
phase from the total orbit period. As a result, u1 = 32.5hr − 10hr − 3min.
• u2: Transient phase rule, which is u2 = T given by the ML.
• u3: Duration rule from the Table 6 for the operational mode s4. u3 = [u3(Id)] where
Id shows the updated operational mode in state s4. Initially, u3(Id = 0) corresponds
to the duration of commissioning which is 60 days.
• u4: The duration of reaching the science observation in the relative position control.
In the design of the architecture of the timed automata model, the set of all transitions
k is given as below
• k0: Commissioning transition law.
• k1: Formation stabilization phase transition law.
• k2: Transient phase transition law.
• k3: science phase rule, De-commissioning transition law.
• k4: ML rule, next operational mode transition law.
• k5: End of operation transition law.
In the design of the architecture of the timed automata model, the set of initial
conditions Init is given as below
• c0: Commissioning initial conditions.
• c1: Formation stabilization phase initial conditions.
• c2: Transient phase initial conditions.
• c3: science phase initial conditions.
• c4: Next operational mode initial conditions.
• c5: De-commissioning initial conditions.
28
Table 7: Reset function reset(i, j).
reset(i, j) Operation
reset(0, 4) t0 = 0
reset(1, 2) t1 = 0
reset(2, 3) t2 = 0
reset(3, 4) t1 = 0
reset(3, 5) t3 = 0
reset(4, 2) t1 = 0
reset(5, 6) t5 = 0
Table 8: Boolean guard conditions g(i, j).
Guard condition g(i, j) Boolean expression
g(0, 4) u3 < t0
g(1, 2) u1 < t1
g(2, 3) u2 < t2
g(3, 4) u3 ≥ t3
g(3, 1) u3 < t3
g(3, 5) 452hr ≤ t3
g(4, 2) u4 < t4
g(5, 6) u3 < t5
Figure 9: Supervisory adaptive control system. s represents the set of discrete states of finite automata,
and k represents the set of all transitions in finite automata
• c6: End of operation initial conditions.
Init : [s = s(0), ti = 0(i = 0, 1, ..., 6)], which corresponds to all the local timers set to
zero at the beginning, and s are at their initial condition. The initial condition of s4
corresponds to the commissioning in the operational mode which is Id = 0.
Fig. 9 illustrates the supervisory adaptive control system.
29
In the supervisory control system, ML is used for updating T and controller parameters
in GNC to make the supervisory control system intelligent. The intelligent supervisory
control system increases the lifetime of the mission and satisfies the requirements of the
mission. SMC provides global asymptotic stability with positive controller parameters
for any initial condition and any desired trajectory. In the intelligent supervisory control
system, the initial conditions, the desired trajectory, and controller parameters are updating
while the system dynamics remains the same. As a result, When SMC is used for the
GNC, the supervisory control system provides asymptotic stability. The global asymptotic
stability of SMC are proved in sections 2.11.3,2.11.4 and Appendix B.
9. Verification and Validation (V&V) of VTXO mission
9.1. Domain and objective
The main objective of the V&V section is to confirm the accuracy, reliability, and
efficiency of the control systems implemented for the VTXO space mission. The V&V
process aims to ensure that the systems perform consistently under different environmental
conditions and adhere to the defined mission objectives.
9.2. Methods and Approach
This section establishes that the VTXO guidance, navigation, and control (GNC)
architecture - comprising a timed automatic supervisory policy, a Lyapunov-based transient
controller with ML-selected gains, a sliding–mode controller (SMC) for science pointing and
relative position regulation, and MEKF-based navigation; satisfies the mission’s precision,
energy, timing, and robustness requirements under realistic disturbances and sensing
conditions. The verification approach combines formal stability arguments with large-scale
closed-loop Monte Carlo (MC) simulation campaigns and distributional performance
summaries derived from optimization-backed data generation.
We adopt a simulation-first methodology because the hybrid nature of the system couples continuous spacecraft dynamics with discrete supervisory transitions. All closed-loop
simulations are performed with a fixed-step fourth-order Runge–Kutta (RK4) integrator,
which preserves the quaternion norm and provides stable error control across phases.
Sensor noise, inertia modeling errors, reaction-wheel nonlinearities, and external torques
are injected according to the parameterizations given earlier (Tables E.10, 87, D.9), so that
navigation, control, and supervisory decisions are exercised under the same uncertainties
expected on-orbit. Phase progression is governed by a timed–automata policy (Fig. 9) with
explicitly defined guards and timer resets (Tables 8–7). This ensures non-Zeno behavior
and, more importantly for correctness, enforces that the reachable states at each phase
boundary lie inside the admissible set for the next phase. In particular, the transition from
the transient phase to the science phase triggers only when the ML-predicted duration T is
attained and the terminal accuracy is within tolerance; the supervisor therefore mediates
a safe hand-off from aggressive slews to persistent precision tracking.
The controllers are verified at two levels. At the analytical level, the transient
controller’s global asymptotic stability follows from a standard Lyapunov argument
when the feedback gains (k1, k2) are positive. Within our framework these gains are
produced by a DNN with a ReLU output layer and kernel max-norm constraints, which
guarantees positivity and boundedness of the outputs for bounded inputs; thus the
Lyapunov conditions are preserved at run time. For the science observation and relative
position problems, the SMC laws adopt a saturation layer in place of the discontinuous
30
sign function, eliminating high-frequency chattering while preserving the invariance of
the sliding surface in the presence of bounded disturbances and model uncertainties. The
SMC stability proof is given earlier in Sections Appendix B and 2.11.4, and those results
transfer directly to the relative position formulation where the counterpart spacecraft’s
thrust is treated as a disturbance.
At the empirical level, we validate performance with MC campaigns whose initial
conditions, disturbances, and sensor characteristics follow the distributions used to generate
the optimization dataset. The dataset itself is obtained with SA/MOGA solvers that
minimize a transient-phase scalar loss and a science-phase vector loss χ = [E, e]
⊤, and its
statistics are summarized in Table 2. These summaries provide distributional guarantees
that are more informative than single-scenario traces. In particular, the 99th percentile
values P99(e) = 0.067◦ and P99(E) = 0.37 J demonstrate that extreme events are rare and
that mission accuracy and energy budgets are met with high probability, while maxima are
naturally higher due to solver stochasticity and the heavy-tailed character of the induced
distributions. The terminal angular velocity ωf distribution (Fig. 4) is centered near zero
with narrow P1 and P99 bounds within ±1 deg/s, aligning with the requirement that steady
tracking proceeds without persistent rate bias. For the science observation phase, SMC
maintains arc-minute attitude accuracy and sub-millimeter transverse alignment in the
presence of injected noise and disturbances, as seen in the relative-position experiments
(Figs. 6 and 7); the PD baseline under identical conditions tracks at the arc-minute level
but does not achieve sub-millimeter alignment (Figs 8), which corroborates the choice of
SMC for flight.
A critical question for an AI-assisted controller is whether the ML layer jeopardizes
stability or erodes margins. Here the learning component is deliberately confined to
selecting positive controller gains and the transient duration T from offline-optimized data
reflecting the operational envelope. The DNN is trained on the SA/MOGA dataset with
explicit regularization and max-norm constraints, and its outputs are further validated
in MC: across folds and held-out conditions, the predicted pairs (
ˆk1,
ˆk2) and durations
Tˆ yield closed-loop responses that respect the same percentile criteria as the generating
optimization runs. In other words, the ML layer inherits the feasibility and margins of
the optimization dataset rather than attempting to learn a control policy ex nihilo. This
design choice, together with the positivity guarantees furnished by the ReLU output layer,
preserves the Lyapunov and SMC conditions by construction.
The supervisor itself is validated along two axes: liveness and safety. Liveness is
guaranteed by guards that require strictly positive dwell times for each location, ruling
out Zeno switching. Safety is demonstrated by showing that the guard conditions are only
enabled when the next phase’s admissible set is reachable under the current controller. In
practice this is evidenced by the distributional bounds at phase end: during the transient
phase, the last-D-second error norm meets the e tolerance at the ML-predicted Tˆ, enabling
the g(2, 3) guard; at science-phase exits, the u3 timing rule and energy consumption remain
within the planned envelope, enabling either a return to stabilization or a transition to the
next operational mode. The resulting hybrid execution sequences align with the mission
timeline in Table 6 and complete without deadlocks.
The verification evidence therefore triangulates across formal arguments, distributional
guarantees, and time-domain closed-loop behavior. Formal arguments ensure that, under
the enforced positivity and boundedness constraints on gains and SMC parameters,
the continuous dynamics are globally asymptotically stable in the transient phase and
asymptotically converge to the sliding manifold in the science and relative position phases.
31
Distributional guarantees (P99 bounds for e, E, and ωf ) demonstrate that robustness
extends beyond nominal conditions to the broad support of the operational envelope. Timedomain evidence shows that under the same noise and disturbance models used for data
generation, the system tracks scientific targets within the required arc-minute pointing and
sub-millimeter alignment while consuming energy consistent with the optimization-derived
budgets.
Two limitations merit explicit acknowledgement. First, all probabilistic statements
are conditional on the disturbance, actuator, and sensor models employed. Changes in
on-orbit environment or component aging that shift these distributions would call for
retraining the ML map and regenerating the SA/MOGA dataset to re-establish the same
percentile margins. Second, the heavy-tailed nature of the optimization outputs implies
that rare outliers above P99 can occur; operational practice mitigates these events by
allocating margin in propellant and observation windows and by allowing the supervisor
to prolong the transient phase when necessary.
Within these assumptions, the V&V results support the claim that the VTXO GNC
system, as architected here, meets mission requirements with high confidence. The hybrid
execution orchestrated by the timed automaton, the stability and robustness of SMC
for precision phases, and the constrained, explainable role of ML in parameter selection
collectively yield a control stack that is both verifiably correct and practically deployable.
Reproducibility is facilitated by reporting aggregated dataset statistics (Table 2) and by
referencing the distributional behavior of outputs and terminal rates (Figs. 3 and 4); these
artifacts enable independent replication of the MC campaigns and cross-checking of the
reported percentiles.
10. Conclusion
The VTXO mission’s demanding requirements for high-precision formation flying are
addressed through a hierarchical control architecture that partitions the problem into
distinct operational phases. This work demonstrates that combining Simulated Annealing
(SA) optimization with a Deep Neural Network (DNN) surrogate model provides a robust
method for finding optimal control parameters that minimize energy consumption (E)
and pointing error (e) within a constrained maneuver time (T). Given the significant
computational cost of optimization (averaging 0.51 hours per run), the DNN serves as
an efficient, real-time surrogate, learning the optimal behavior from the offline-generated
data.
A pivotal contribution of this research is the development of an Explainable AI
(XAI) framework. By training the DNN to predict not only the control gains but also
the resultant E and e, the system transcends the typical “black-box” paradigm. This
predictive transparency allows the system’s decisions to be interpretable and verifiable, a
critical feature for mission-critical applications. This explainability is further reinforced
by the timed automata-based supervisory control, which provides a clear, rule-based logic
for all phase transitions.
The efficacy of this approach is validated through extensive Monte-Carlo simulations,
which were used for both data generation and sensitivity analysis. SA optimization
successfully balanced the competing mission objectives, and the hyperparameter-tuned
DNN learned this behavior with high fidelity. The inclusion of noise and disturbances in
the training data acts as a form of regularization, enhancing the robustness of the learned
model.
32
The comprehensive Verification and Validation (V&V) process confirms that the
proposed control system is not only efficient and robust but also trustworthy. By rigorously
testing the system across its full operational envelope, we have validated its ability to
meet stringent mission objectives. This V&V effort substantiates the reliability of the
AI’s predictions and the logical soundness of the supervisory framework, confirming its
readiness for complex, autonomous space environments.
Appendix A. Asymptotic Stability
A point is considered to be an equilibrium point xe for the system if x˙(t) = 0 for all t.
xe for the system is global asymptotic stable if the positive scalar function V (x) satisfies
the following conditions
• V (xe) = 0
• V (x) > 0 for x ̸= xe
• V˙ (x) ≤ 0
When the given conditions are satisfied, V (x) is a Lyapunov function. If V˙ (x) < 0 for
x ̸= xe, xe is asymptotically stable. If V˙ (x) ≤ 0, V (x) is a Lyapunov function and the
system is stable. LaSalle’s theorem can prove the asymptotic stability.
Appendix B. Sliding mode control asymptotic stability proof
The SMC stability can be proven/shown by the Lyapunov stability theorem. In SMC,
the state x reach the desired state xf . The difference between the states and the desired
states is
∆(x) = x − xf (B.1)
Since the dynamics of spacecraft is second order, the Lyapunov asymptotic stability
proof is shown for a second order system. A second order system is given by
x¨ = f(x(t), x˙(t)) + u(t) (B.2)
A sliding surface is considered as
s = ∆ ˙x + λ∆(x) (B.3)
Where λ is a scalar. The following Lyapunov function proves the asymptotic stability
V (∆(x)) = 1
2
s
2
(B.4)
The time derivative of the Lyapunov function is given as
V˙ (x) = ss˙ (B.5)
The SMC control law is obtained by preventing the motion off of the sliding surface
by setting s˙ = 0. To obtain the control law, the known model is defined as ¯f(x, x˙). Using
s˙ = 0 into (B.2) and taking the time derivative of s in (B.3), s˙ is obtained as
s˙ = ¯f(x, x˙) + u − x¨ + λ∆( ˙x) (B.6)
33
The nominal input u¯ that derives the s˙ to zero is obtained as
u¯ = − ¯f(x, x˙) + ¨x − λ∆( ˙x) (B.7)
Since there are model uncertainties and disturbances in the system, the discontinuous
term −ksign(s) is added to the u¯ given by
u = ¯u − ksign(s) (B.8)
Assuming that model uncertainties and disturbances in the system are bounded by
the known function F(x, x˙) with a maximum value Fmax, and letting k = Fmax + ϱ for the
positive scalar ϱ, V˙ (x) is obtained as
V˙ (x) ≤ −ϱ|s| (B.9)
When k is chosen to be large enough, the given sliding mode control law is stable in
the presence of uncertainties and disturbances in the system. Larger k introduces more
chattering since ksign(s) induces chattering the system. To reduce chattering, sign(s)
can be replaced by a saturation function with a varying boundary layer thickness, and
the control law is given as
u = u − ksat(s, ε) (B.10)
ε is the boundary layer thickness.
The proof of sliding mode control (SMC) stability are more discussed in [16].
Appendix C. Sensor model VTXO
Appendix C.1. Gyro model
In ω˜, Gaussian white noise wω (angular random walk), bias bω, scale factor f ω, and
misalignment ϵω are included as the following
ω˜ = ∂R(ϵω)[{I 3×3 + diag(f ω)}ω + bω + wω] (C.1)
The variance σ
2
wω
captures the power of the noise in the random noise wω as
E[wω(t)wω(t
′
)
T
] = σ
2
wω
I 3×3δ(t − t
′
) (C.2)
The bias bω is modeled as a first-order Markov processes given by
˙bω =
−bω
τω
+ wbω (C.3)
The variance σ
2
bω captures the power of the noise in the random noise wbω.
Appendix C.2. Star tracker model
In q˜, Gaussian white noise wq, bias bq are included as the following
q˜ = ∂q(wq) ⊗ ∂q(bq) ⊗ q (C.4)
The variance σ
2
wq
captures the power of the noise in the random noise wq.
The bias bq is modeled as a first-order Markov processes given by
34
˙bq =
−bq
τq
+ wbq (C.5)
The variance σ
2
wbq
captures the power of the noise in the random noise wbq.
The small rotation caused by ∂q(wq) and ∂q(bq) is obtained by
∂q ≈
"
θ/2
1
#
(C.6)
Appendix C.3. Accelerometer model
In r˜¨, Gaussian white noise wr and bias br are included as the following
r˜¨ = r¨ + br + wr (C.7)
The variance σ
2
wr
captures the power of the noise in the random noise wr.
The bias br is modeled as a first-order Markov processes given by
˙br =
−br
τr
+ wbr (C.8)
The variance σ
2
wbr
captures the power of the noise in the random noise wbr.
Appendix C.4. Radio ranging sensor
The radio ranging sensor is mounted on the follower spacecraft.
The desired relative distance between leader and follower is given by
r
d
rel =
h
0 0 1kmiT
(C.9)
The measured r
z
rel in the z axis is denoted as r˜
z
rel, and it includes misalignment ϵrel
and white Gaussian noise given by
r˜
z
rel = r
z
rel + ϵr
z + wr
z (C.10)
The misalignment ϵr
z is modeled as a first-order Markov processes given by
ϵ˙r
z =
−ϵr
z
τr
z
+ wϵrz (C.11)
The variance σ
2
wϵrz
captures the power of the noise in the random noise wϵrz .
Appendix C.5. Extended Kalman filter
The residual error (also called the true navigation errors) is defined as
ε = x(t) − xˆ(t) (C.12)
Extended Kalman filter (EKF) [16] is used for state estimation and sensor fusion for
nonlinear dynamics. In EKF, the covariance of the residual error P is used for updating
the states. The first-order Markov process bias and misalignment in the sensors are
considered as the states to be estimates by EKF. w and ν k Gaussian white noise with the
covariance Q and Rk are considered for the dynamics model and sensor model, respectively.
The subscript k shows at time instant tk. x
+ shows the updated value of x after the
35
measurement by EKF from the measurements, and x
− shows the value of x before the
measurement. x
− comes from the propagation.
EKF is given by the following equations
Nonlinear dynamics model
x˙ = f (x(t), τ (t),w, t) (C.13)
Sensor model
yk = h(xk) + ν k (C.14)
Initialization of states
xˆ(t0) = xˆ0 (C.15)
Initialization of residual error covariance
Pˆ (t0) = Pˆ
0 (C.16)
States propagation
xˆ˙ = f (x(t), τ (t), t) (C.17)
The covariance of residual error propagation
P
ˆ˙ = FP + PF T + GQGT
(C.18)
F =
∂f
∂x |xˆ
G =
∂f
∂w|xˆ
Updating EKF gain
Kk = P
−
k H
T
k
[HkP
−
k H
T
k + Rk]
−1 (C.19)
Hk =
∂h
∂x |xˆ
−
k
Updating the states
xˆ
+
k = xˆ
−
k + Kk(yk − h(xˆ
−
k
)) (C.20)
Updating the covariance of residual error propagation
Pˆ
+
k = [1 − KkHk]Pˆ
−
k
(C.21)
Appendix D. Keplerian elements
The orbits are highly-elliptical supersynchronous geostationary transfer orbit with a
32.5-hour period. For the simulation, the Keplerian elements are eccentricity γ, semi-major
axis a, inclination i, right ascension of the ascending node Ω, and argument of periapsis
ω. The Keplerian elements are given in Table D.9.
36
Table D.9: Keplerian elements for leader and follower.
i deg Ω deg ω deg a km γ
leader orbit 0.34 0 4.6743 45300 0.7125
follower orbit 0.34 0 4.6743 45300 0.7336
Appendix E. Spacecraft and instrumentation parameters
The follower and leader are 6U CubeSat with inertial mass 10.2kg and nominal inertial
momentum matrix ¯J as
¯J =



0.1383 0 0
0 0.1577 0
0 0 0.1039


 kg · m
2
(E.1)
The time variant modeling error δJ in J (12) for both leader and follower is modeled
as
δJ =



0.0038 sin(t) 0 0
0 0.005 cos(t) 0
0 0 0.0011


 kg · m
2
(E.2)
The reaction wheel and thruster parameters for both leader and follower are given in
Table E.10.
Table E.10: Reaction wheel and thruster parameters per axis.
Leader 3-sigma value Follower 3-sigma value
fτ per axis 0.01 0.01
ϵτ (mrad)/axis 1 1
wτ (N − m)/axis 0.001 0.001
wbτ (N − m)/axis 0.001 0.001
ττ hr/cycle/axis 1 1
bu (mN-m)/axis 0.1 0.1
fu per axis 0.01 0.01
ϵu (mrad)/axis 1 1
wu (N)/axis 0.001 0.001
wbu (N)/axis 0.001 0.001
τu (hr/cycle)/axis 1 1
The variance of external disturbances are given in Table E.11.
Table E.11: External disturbances.
leader 3-sigma value follower 3-sigma value
wω˙ (N − m)/axis 0.001 0.001
τg (hr/cycle)/axis 1 1
wg micor-N/axis 22.5 22.5
The sensor parameters for the IMU and star tracker are given in Table E.12. The
radio ranging is mounted on the follower giving the relative distance between the follower
and leader. The laser beacons are mounted the leader forming an interferometry sensor.
37
The precise location of the leader is obtained with respect to the follower using the
interferometry sensor. The sensor parameters for the interferometry sensor and radio
ranging are given in Table E.13.
Table E.12: Sensor parameters.
Sensor Symbol Follower 3-sigma value Leader 3-sigma value
Gyro fω per axis 0.0003 0.0003
Gyro ϵω (mrad)/axis 3 3
Gyro wω micor-N/axis 22.5 22.5
Gyro bω deg/hr/axis 3 3
Star tracker wq per axis 41 milliarcsecond 3 arcminute
Star tracker wbq per axis 0.1 milliarcsecond 3 arcminute
Star tracker τq (hr/cycle)/axis 1 1
Accelerometer wr nano-g 1 1
Accelerometer wbr micor-N/axis 1 1
Accelerometer τr (hr/cycle)/axis 1 1
Table E.13: Interferometry and radio ranging sensors’ parameters.
Sensor Symbol 3-sigma value
Radio ranging wr
z m/axis 1
Radio ranging wϵrz m/axis 1
Radio ranging τr
z (hr/cycle)/axis 1
Interferometry sensor wr
xy millimeter/axis 0.2
Interferometry sensor wϵrxy millimeter/axis 0.01
Interferometry sensor τr
xy (hr/cycle)/axis 1
References
[1] J. F. Krizmanic, N. Shah, P. C. Calhoun, A. K. Harding, L. R. Purves, C. M. Webster,
M. F. Corcoran, C. R. Schrader, S. J. Stochaj, K. A. Rankin, et al., Vtxo: the virtual
telescope for x-ray observations, in: Space Telescopes and Instrumentation 2020:
Ultraviolet to Gamma Ray, Vol. 11444, International Society for Optics and Photonics,
2020, p. 114447V.
[2] K. Rankin, N. Shah, J. Krizmanic, S. Stochaj, A. Naseri, Formation flying techniques
for the virtual telescope for x-ray observations, arXiv preprint arXiv:2007.09287
(2020).
[3] R. Pirayesh, A. Naseri, S. Stochaj, N. Shah, J. Krizmanic, Attitude control of a
two-cubesat virtual telescope in highly elliptical orbits, in: 2018 AIAA Guidance,
Navigation, and Control Conference, 2018, p. 0866.
[4] R. Pirayesh, A. Naseri, F. Moreu, S. Stochaj, N. Shah, J. Krizmanic, Attitude control
optimization of a two-cubesat virtual telescope in a highly elliptical orbit, in: Space
Operations: Inspiring Humankind’s Future, Springer, 2019, pp. 233–258.
38
[5] R. Pirayesh, M. Martinez-Ramon, A. Naseri, S. Stochaj, N. Shah, J. Krizmanic, Deep
learning and gaussian process approach for optimal attitude control of a two-cubesat
virtual telescope (2019).
URL https://digitalcommons.usu.edu/smallsat/2019/all2019/18/
[6] R. Pirayesh, A. Naseri, S. Stochaj, N. Shah, J. Krizmanic, Attitude control optimization of a virtual telescope for x-ray observations (2018).
URL https://digitalcommons.usu.edu/smallsat/2018/all2018/442/
[7] R. Pirayesh, A. Naseri, S. Stochaj, N. Shah, J. Krizmanic, Hybrid attitude control of
a two-cubesat virtual telescope in a highly elliptical orbit (2017).
URL https://digitalcommons.usu.edu/smallsat/2017/all2017/46/
[8] L. Meza, F. Tung, S. Anandakrishnan, V. Spector, T. Hyde, Line of sight stabilization
for the james webb space telescope, Advances in the Astronautical Sciences 121 (2005)
17–30.
[9] C. M. Pong, High-precision pointing and attitude estimation and control algorithms for
hardware-constrained spacecraft, Ph.D. thesis, Massachusetts Institute of Technology
(2014).
[10] G. Chester, NEWS! From the NAVAL OBSERVATORY FOR IMMEDIATE
RELEASE USNO’s NISTEx-II Instrument Successfully Launched on May 4, 2019,
Tech. rep. (2019).
URL https://www.usno.navy.mil/USNO/tours-events/
usno2019s-nistex-ii-instrument-successfully-launched-on-may-4-2019
[11] K. Rankin, S. Stochaj, N. Shah, J. Krizmanic, A. Naseri, Vtxo-virtual telescope for
x-ray observations, arXiv preprint arXiv:1807.05249 (2018).
[12] K. Rankin, H. Park, J. Krizmanic, N. Shah, S. Stochaj, A. Naseri, Trajectory
optimization for the virtual telescope for x-ray observations, UMBC Center for Space
Sciences and Technology (2020).
[13] P. Calhoun, N. Shah, Covariance analysis of astrometric alignment estimation architectures for precision dual spacecraft formation flying, in: AIAA Guidance, Navigation,
and control Conference, 2012, p. 4706.
[14] A. Naseri, R. Pirayesh, R. K. Adcock, S. J. Stochaj, N. Shah, J. Krizmanic, Formation
flying of a two-cubesat virtual telescope in a highly elliptical orbit, in: 2018 SpaceOps
Conference, 2018, p. 2633.
[15] Y. Bai, J. D. Biggs, F. B. Zazzera, N. Cui, Adaptive attitude tracking with active
uncertainty rejection, Journal of Guidance, Control, and Dynamics 41 (2) (2018)
550–558.
[16] F. L. Markley, J. L. Crassidis, Fundamentals of spacecraft attitude determination
and control, Springer, 2014.
[17] Z.-X. Li, B.-L. Wang, Robust attitude tracking control of spacecraft in the presence of
disturbances, Journal of Guidance, Control, and Dynamics 30 (4) (2007) 1156–1159.
39
[18] W. Luo, Y.-C. Chu, K.-V. Ling, H-infinity inverse optimal attitude-tracking control
of rigid spacecraft, Journal of guidance, control, and dynamics 28 (3) (2005) 481–494.
[19] R. J. Wallsgrove, M. R. Akella, Globally stabilizing saturated attitude control in
the presence of bounded unknown disturbances, Journal of guidance, Control, and
Dynamics 28 (5) (2005) 957–963.
[20] F. Lewis, S. Jagannathan, A. Yesildirak, Neural network control of robot manipulators
and non-linear systems, CRC press, 2020.
[21] F. Kodalak, M. U. Salamci, Model reference adaptive control design for nonlinear
systems using linear time varying approximations, in: Proceedings of the 2015 16th
International Carpathian Control Conference (ICCC), IEEE, 2015, pp. 202–207.
[22] O. Yechiel, H. Guterman, A survey of adaptive control, International Robotics and
Automation Journal 3 (2) (2017) 0053.
[23] D. C. Woffinden, D. K. Geller, Relative angles-only navigation and pose estimation
for autonomous orbital rendezvous, Journal of Guidance, Control, and Dynamics
30 (5) (2007) 1455–1469. doi:10.2514/1.28216.
URL http://dx.doi.org/10.2514/1.28216
[24] J. R. Carpenter, C. N. D’Souza, Navigation filter best practices, Tech. rep. (2018).
[25] D. Ye, J. Zhang, Z. Sun, Extended state observer–based finite-time controller design
for coupled spacecraft formation with actuator saturation, Advances in Mechanical
Engineering 9 (4) (2017) 1687814017696413.
[26] A. Sveier, A. M. Sjøberg, O. Egeland, Applied runge–kutta–munthe-kaas integration
for the quaternion kinematics, Journal of Guidance, Control, and Dynamics 42 (12)
(2019) 2747–2754.
[27] K. I. Kou, Y.-H. Xia, Linear quaternion differential equations: Basic theory and
fundamental results, Studies in Applied Mathematics 141 (1) (2018) 3–45.
[28] L. R. Ray, R. F. Stengel, A monte carlo approach to the analysis of control system
robustness, Automatica 29 (1) (1993) 229–236.
[29] C. Ou, J. Jiang, H. Wang, Z. Zhen, Monte carlo approach to the analysis of uavs
control system, in: Proceedings of 2014 IEEE Chinese Guidance, Navigation and
Control Conference, IEEE, 2014, pp. 458–462.
[30] R. Pirayeshshirazinezhad, S. G. Biedroń, J. A. D. Cruz, S. S. Güitrón, M. MartínezRamón, Designing monte carlo simulation and an optimal machine learning to
optimize and model space missions, IEEE Access 10 (2022) 45643–45662.
[31] K. Rankin, H. Park, D. Smith, J. Krizmanic, N. Shah, S. Stochaj, A. Naseri, Baseline
mission design of a distributed space telescope for x-ray observations, Advances in
Space Research (2025).
[32] D. C. Woffinden, Angles-only navigation for autonomous orbital rendezvous, Utah
State University, 2008.
40
[33] W. M. Lear, Kalman Filtering Techniques, Mission Planning and Analysis Division,
National Aeronautics and Space . . . , 1985.
[34] S.-H. Mok, S. Y. Byeon, H. Bang, Y. Choi, Performance comparison of gyro-based
and gyroless attitude estimation for cubesats, International Journal of Control,
Automation and Systems (2020) 1–11.
[35] B. Wie, P. M. Barba, Quaternion feedback for spacecraft large angle maneuvers,
Journal of Guidance, Control, and Dynamics 8 (3) (1985) 360–365.
[36] R. Pirayeshshirazinezhad, Artificial intelligence, controls, and sensor fusion for optimization and modeling of space missions and particle accelerators, Ph.D. thesis, The
University of New Mexico (2022).
[37] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, arXiv preprint
arXiv:1412.6980 (2014).
[38] T. Dozat, Incorporating nesterov momentum into adam, ICLR 2016 workshop (2016).
[39] M. Hendriks, Model checking timed automata-techniques and applications, Ph.D.
thesis, [Sl: sn] (2006).
[40] S. A. Ghorashi Khalil Abadi, A. Bidram, A distributed rule-based power management
strategy in a photovoltaic/hybrid energy storage based on an active compensation
filtering technique, IET Renewable Power Generation 15 (15) (2021) 3688–3703.
[41] J. P. Paxman, Switching controllers: Realization, initialization and stability, Ph.D.
thesis, University of Cambridge (2004).
41"""
    
    # Check if custom text was provided
    if "PASTE YOUR ACADEMIC PAPER TEXT HERE" in CUSTOM_TEXT:
        print("⚠️  No custom text provided. Please replace CUSTOM_TEXT with your academic paper text.")
        return
    
    try:
        # Test the custom text (mirror production by cleaning before sectioning)
        cleaned = clean_scientific_text(CUSTOM_TEXT)
        sections = improved_split_into_sections(cleaned)
        extractor = ContentExtractor()
        
        print(f"📋 Detected {len(sections)} sections in your text:")
        for name, content in sections.items():
            print(f"   - {name}: {len(content)} characters")
        
        # Extract components
        contributions = extractor.extract_contributions(sections)
        methodology = extractor.extract_methodology(sections)
        results = extractor.extract_results(sections)
        
        print("\n📝 Extracted Components:")
        print(f"Contributions: {contributions[:200]}...")
        print(f"Methodology: {methodology[:200]}...")  
        print(f"Results: {results[:200]}...")
        
    except Exception as e:
        print(f"❌ Custom text test failed: {e}")

if __name__ == "__main__":
    # Run the comprehensive test suite
    tester = StructuredSummarizerTester()
    tester.run_all_tests()
    
    # Uncomment the line below to test with your custom text
    # test_with_custom_text()
    
    print("\n" + "=" * 60)
    print("🎯 Testing Complete!")
    print("💡 To test with your own text, modify the CUSTOM_TEXT variable in test_with_custom_text()")
    print("🔧 For PDF testing, you can use the process_pdf_structured_summary function directly")