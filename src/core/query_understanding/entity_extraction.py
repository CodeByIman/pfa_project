from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

try:
    import spacy
    has_spacy = True
except Exception:
    has_spacy = False

# Listes étendues pour la détection d'entités dans les requêtes de recherche scientifique

DOMAIN_KEYWORDS = [
    # Computer Vision & Image Processing - Extended
    'computer vision', 'image processing', 'object detection', 'image segmentation', 'face recognition',
    'optical character recognition', 'ocr', 'image classification', 'image generation', 'super resolution',
    'style transfer', 'image inpainting', 'depth estimation', 'pose estimation', 'action recognition',
    'video analysis', 'motion detection', 'tracking', 'visual slam', 'stereo vision', '3d reconstruction',
    'medical imaging', 'satellite imagery', 'autonomous driving', 'surveillance', 'augmented reality',
    'virtual reality', 'mixed reality', 'photogrammetry', 'image restoration', 'denoising',
    'image enhancement', 'histogram equalization', 'contrast enhancement', 'edge detection',
    'feature extraction', 'keypoint detection', 'descriptor matching', 'image registration',
    'template matching', 'optical flow', 'scene understanding', 'visual reasoning',
    'visual question answering', 'image captioning', 'visual grounding', 'visual navigation',
    'structure from motion', 'bundle adjustment', 'epipolar geometry', 'homography',
    'camera calibration', 'lens distortion', 'panoramic imaging', 'light field imaging',
    'computational photography', 'hdr imaging', 'tone mapping', 'image stitching',
    'morphological operations', 'watershed segmentation', 'region growing', 'active contours',
    'level sets', 'graph cuts', 'random walker', 'grabcut', 'meanshift', 'superpixels',
    'slic', 'quickshift', 'felzenszwalb', 'texture analysis', 'gabor filters', 'lbp',
    'local binary patterns', 'haralick features', 'glcm', 'histogram of oriented gradients',
    'hog', 'sift', 'surf', 'orb', 'brief', 'freak', 'akaze', 'kaze',
    'corner detection', 'harris corner', 'shi-tomasi', 'fast', 'blob detection',
    'difference of gaussians', 'laplacian of gaussian', 'mexican hat', 'ridge detection',
    'hough transform', 'circle detection', 'line detection', 'ellipse detection',
    'contour analysis', 'shape analysis', 'fourier descriptors', 'moment invariants',
    'zernike moments', 'hu moments', 'geometric moments', 'central moments',
    
    # Medical Imaging - Ultra Extended
    'medical', 'healthcare', 'radiology', 'genomics', 'bioinformatics', 'computational biology',
    'drug discovery', 'medical imaging', 'clinical decision support', 'electronic health records',
    'precision medicine', 'personalized medicine', 'epidemiology', 'public health', 'telemedicine',
    'medical diagnosis', 'pathology', 'oncology', 'cardiology', 'neurology', 'dermatology',
    'ophthalmology', 'medical image analysis', 'biomedical signal processing', 'physiological monitoring',
    'radiomics', 'quantitative imaging', 'imaging biomarkers', 'computer-aided diagnosis', 'cad',
    'computer-aided detection', 'screening', 'early detection', 'disease progression',
    'treatment planning', 'surgical planning', 'image-guided surgery', 'interventional radiology',
    'robotic surgery', 'minimally invasive surgery', 'surgical simulation', 'anatomical modeling',
    'biomechanical modeling', 'finite element analysis', 'computational fluid dynamics', 'cfd',
    'hemodynamics', 'blood flow', 'cardiac modeling', 'electrophysiology', 'ecg', 'eeg',
    'emg', 'eog', 'meg', 'fmri', 'pet', 'spect', 'ct', 'mri', 'ultrasound',
    'x-ray', 'mammography', 'angiography', 'fluoroscopy', 'endoscopy', 'microscopy',
    'histopathology', 'cytology', 'digital pathology', 'whole slide imaging', 'telepathology',
    'molecular imaging', 'multimodal imaging', 'image fusion', 'registration', 'atlas-based',
    'population studies', 'longitudinal studies', 'cohort analysis', 'biostatistics',
    'clinical trials', 'randomized controlled trials', 'rct', 'systematic review', 'meta-analysis',
    'evidence-based medicine', 'ebm', 'quality assurance', 'quality control', 'standardization',
    'interoperability', 'dicom', 'hl7', 'fhir', 'pacs', 'ris', 'his', 'emr', 'ehr',
    'healthcare informatics', 'clinical informatics', 'nursing informatics', 'pharmacy informatics',
    'dental informatics', 'veterinary informatics', 'public health informatics', 'consumer health',
    'health literacy', 'patient engagement', 'shared decision making', 'patient-reported outcomes',
    'quality of life', 'health economics', 'cost-effectiveness', 'health technology assessment',
    'regulatory affairs', 'fda approval', 'ce marking', 'iso standards', 'iec standards',
    'medical device', 'in vitro diagnostics', 'ivd', 'laboratory medicine', 'point-of-care',
    'bedside testing', 'rapid diagnostics', 'biosensors', 'wearable devices', 'mhealth',
    'digital health', 'digital therapeutics', 'artificial pancreas', 'closed-loop systems',
    
    # Federated Learning & Distributed Computing - Comprehensive
    'federated learning', 'distributed learning', 'decentralized learning', 'collaborative learning',
    'privacy-preserving learning', 'secure computation', 'edge learning', 'fog computing',
    'federated optimization', 'federated averaging', 'fedavg', 'distributed optimization',
    'consensus algorithms', 'byzantine fault tolerance', 'differential privacy', 'homomorphic encryption',
    'secure multiparty computation', 'smc', 'private set intersection', 'oblivious transfer',
    'zero-knowledge proofs', 'blockchain', 'distributed ledger', 'smart contracts',
    'peer-to-peer learning', 'swarm learning', 'gossip protocols', 'epidemic algorithms',
    'federated medical imaging', 'distributed healthcare ai', 'privacy-preserving healthcare',
    'secure medical ai', 'collaborative medical research', 'multi-institutional learning',
    'cross-silo federated learning', 'cross-device federated learning', 'horizontal federated learning',
    'vertical federated learning', 'federated transfer learning', 'personalized federated learning',
    'clustered federated learning', 'hierarchical federated learning', 'asynchronous federated learning',
    'communication-efficient federated learning', 'compression', 'quantization', 'sparsification',
    'gradient compression', 'model compression', 'knowledge distillation', 'pruning',
    'federated analytics', 'federated statistics', 'federated sql', 'federated databases',
    'split learning', 'splitfed', 'distributed deep learning', 'parameter servers',
    'allreduce', 'ring-allreduce', 'tree-allreduce', 'butterfly mixing', 'local sgd',
    'elastic averaging', 'downpour sgd', 'admm', 'proximal methods', 'primal-dual methods',
    
    # Natural Language Processing - Ultra Extended
    'nlp', 'natural language processing', 'text mining', 'sentiment analysis', 'machine translation',
    'question answering', 'text summarization', 'named entity recognition', 'ner', 'part-of-speech tagging',
    'syntactic parsing', 'semantic parsing', 'coreference resolution', 'relation extraction',
    'information extraction', 'text generation', 'language modeling', 'dialogue systems', 'chatbots',
    'conversational ai', 'text classification', 'topic modeling', 'document similarity', 'paraphrasing',
    'word embeddings', 'sentence embeddings', 'multilingual processing', 'cross-lingual',
    'computational linguistics', 'natural language understanding', 'nlu', 'natural language generation',
    'nlg', 'speech recognition', 'automatic speech recognition', 'asr', 'speech synthesis',
    'text-to-speech', 'tts', 'voice conversion', 'speaker recognition', 'speaker verification',
    'speaker diarization', 'voice activity detection', 'vad', 'speech enhancement',
    'noise reduction', 'echo cancellation', 'beamforming', 'source separation',
    'phoneme recognition', 'acoustic modeling', 'language modeling', 'pronunciation modeling',
    'prosody', 'intonation', 'stress', 'rhythm', 'accent', 'dialect', 'sociolinguistics',
    'psycholinguistics', 'neurolinguistics', 'cognitive linguistics', 'corpus linguistics',
    'lexicography', 'terminoloy', 'ontologies', 'knowledge graphs', 'semantic web',
    'rdf', 'owl', 'sparql', 'linked data', 'knowledge representation', 'reasoning',
    'logic programming', 'expert systems', 'rule-based systems', 'fuzzy logic',
    'word sense disambiguation', 'wsd', 'semantic role labeling', 'srl', 'semantic similarity',
    'textual entailment', 'natural language inference', 'nli', 'reading comprehension',
    'machine reading', 'open information extraction', 'oie', 'distant supervision',
    'active learning', 'semi-supervised learning', 'unsupervised learning', 'self-supervised learning',
    'contrastive learning', 'metric learning', 'few-shot learning', 'zero-shot learning',
    'in-context learning', 'prompt engineering', 'prompt tuning', 'prefix tuning',
    'adapter tuning', 'lora', 'low-rank adaptation', 'parameter-efficient fine-tuning',
    'instruction tuning', 'rlhf', 'reinforcement learning from human feedback',
    'constitutional ai', 'red teaming', 'adversarial training', 'robustness',
    'fairness', 'bias', 'toxicity', 'hate speech', 'misinformation', 'fact-checking',
    'explainability', 'interpretability', 'attention visualization', 'probing',
    'linguistic analysis', 'syntactic analysis', 'morphological analysis', 'phonological analysis',
    
    # Vision Transformers & Modern Architectures
    'vision transformers', 'vit', 'medical vision transformers', 'transformer segmentation',
    'swin transformer', 'pyramid vision transformer', 'pvt', 'deit', 'data-efficient image transformers',
    'cait', 'class-attention in image transformers', 'twins', 'spatially separable self-attention',
    'crossvit', 'cross-attention multi-scale vision transformer', 'beit', 'bidirectional encoder',
    'mae', 'masked autoencoders', 'simvit', 'simple vision transformers', 'convnext',
    'swin-unet', 'medical transformer', 'transunet', 'medical image transformer',
    'segmenter', 'setr', 'segmentation transformer', 'max-deeplab', 'axial attention',
    'linear attention', 'performer', 'linformer', 'efficient transformers', 'sparse attention',
    'longformer', 'bigbird', 'reformer', 'routing transformer', 'synthesizer',
    'switch transformer', 'mixture of experts', 'moe', 'gshard', 'glam', 'palm',
    'pathways', 'flamingo', 'coca', 'clip', 'align', 'florence', 'beit-3',
    'unilm', 'layoutlm', 'dit', 'document image transformer', 'donut',
    'pix2struct', 'matcha', 'unified-io', 'gpt-4v', 'llava', 'minigpt-4',
    'blip', 'blip-2', 'instructblip', 'mplug', 'mplug-owl', 'qwen-vl',
    'internvl', 'cogvlm', 'llava-next', 'mobilevitvit', 'efficientvit',
    'edgevit', 'levit', 'mobilevit', 'nextvit', 'poolformer', 'metaformer',
    
    # Segmentation - Ultra Comprehensive
    'image segmentation', 'semantic segmentation', 'instance segmentation', 'panoptic segmentation',
    'medical image segmentation', 'brain tumor segmentation', '3d segmentation', 'volumetric segmentation',
    'organ segmentation', 'lesion segmentation', 'tumor segmentation', 'cell segmentation',
    'nuclei segmentation', 'vessel segmentation', 'airway segmentation', 'cardiac segmentation',
    'lung segmentation', 'liver segmentation', 'kidney segmentation', 'prostate segmentation',
    'brain segmentation', 'white matter', 'gray matter', 'cerebrospinal fluid', 'csf',
    'cortical segmentation', 'subcortical segmentation', 'hippocampus segmentation', 'lesion detection',
    'multiple sclerosis', 'stroke segmentation', 'ischemic stroke', 'hemorrhagic stroke',
    'tumor grading', 'glioma segmentation', 'meningioma', 'glioblastoma', 'low-grade glioma',
    'high-grade glioma', 'enhancing tumor', 'tumor core', 'whole tumor', 'peritumoral edema',
    'necrosis', 'non-enhancing tumor', 'active tumor', 'treatment effects', 'pseudoprogression',
    'radiation necrosis', 'recurrence', 'progression', 'response assessment', 'rano criteria',
    'watershed segmentation', 'region growing', 'split and merge', 'active contours',
    'snakes', 'level sets', 'geodesic active contours', 'chan-vese', 'mumford-shah',
    'graph cuts', 'normalized cuts', 'random walker', 'power watershed', 'grabcut',
    'lazy snapping', 'interactive segmentation', 'scribble-based', 'click-based',
    'weakly supervised segmentation', 'semi-supervised segmentation', 'unsupervised segmentation',
    'domain adaptation', 'cross-domain segmentation', 'sim-to-real', 'synthetic data',
    'data augmentation', 'geometric augmentation', 'photometric augmentation', 'mixup',
    'cutmix', 'mosaic', 'copy-paste', 'weak labels', 'noisy labels', 'label noise',
    'label smoothing', 'pseudo-labeling', 'self-training', 'co-training', 'tri-training',
    'multi-view learning', 'consensus-based', 'disagreement-based', 'diversity-based',
    
    # Privacy & Security - Extended
    'cybersecurity', 'network security', 'intrusion detection', 'malware detection',
    'privacy preservation', 'differential privacy', 'local differential privacy', 'global differential privacy',
    'epsilon-differential privacy', 'delta-differential privacy', 'renyi differential privacy',
    'concentrated differential privacy', 'approximate differential privacy', 'pure differential privacy',
    'composition theorems', 'privacy amplification', 'subsampling', 'shuffling', 'gaussian mechanism',
    'laplace mechanism', 'exponential mechanism', 'sparse vector technique', 'svt',
    'private aggregation of teacher ensembles', 'pate', 'dpsgd', 'differentially private sgd',
    'moment accountant', 'rdp accountant', 'privacy budget', 'privacy loss', 'utility-privacy tradeoff',
    'homomorphic encryption', 'fully homomorphic encryption', 'fhe', 'somewhat homomorphic encryption',
    'she', 'leveled homomorphic encryption', 'bootstrapping', 'noise management', 'packing',
    'batching', 'simd operations', 'ciphertext operations', 'plaintext operations',
    'key generation', 'encryption', 'decryption', 'evaluation', 'circuit depth',
    'multiplicative depth', 'noise budget', 'parameter selection', 'security level',
    'lattice-based cryptography', 'ring learning with errors', 'rlwe', 'learning with errors',
    'lwe', 'short integer solution', 'sis', 'ntru', 'bfv', 'ckks', 'tfhe', 'fhew',
    'secure multiparty computation', 'mpc', 'secret sharing', 'shamir secret sharing',
    'replicated secret sharing', 'additive secret sharing', 'boolean secret sharing',
    'garbled circuits', 'yao garbled circuits', 'oblivious transfer', 'ot', 'ot extension',
    'private information retrieval', 'pir', 'oblivious ram', 'oram', 'private set intersection',
    'psi', 'private set union', 'psu', 'secure aggregation', 'secure sum', 'secure averaging',
    'byzantine robustness', 'byzantine fault tolerance', 'honest majority', 'dishonest majority',
    'semi-honest model', 'malicious model', 'covert model', 'universal composability', 'uc',
    'zero-knowledge proofs', 'zk-snarks', 'zk-starks', 'bulletproofs', 'plonk', 'sonic',
    'marlin', 'aurora', 'ligero', 'stark', 'fri', 'polynomial commitment schemes',
    'trusted setup', 'universal trusted setup', 'transparent setup', 'preprocessing', 'proving time',
    'verification time', 'proof size', 'soundness', 'completeness', 'zero-knowledge property',
    
    # Reinforcement Learning - Extended
    'reinforcement learning', 'rl', 'deep reinforcement learning', 'drl', 'markov decision process',
    'mdp', 'partially observable mdp', 'pomdp', 'multi-agent reinforcement learning', 'marl',
    'cooperative multi-agent', 'competitive multi-agent', 'mixed-motive', 'game theory',
    'nash equilibrium', 'correlated equilibrium', 'stackelberg equilibrium', 'mechanism design',
    'auction theory', 'voting theory', 'social choice', 'fair division', 'matching theory',
    'value function', 'action-value function', 'q-function', 'state-value function', 'v-function',
    'policy', 'stochastic policy', 'deterministic policy', 'optimal policy', 'policy gradient',
    'actor-critic', 'advantage function', 'temporal difference', 'td learning', 'td error',
    'eligibility traces', 'n-step methods', 'monte carlo methods', 'bootstrapping',
    'on-policy', 'off-policy', 'importance sampling', 'behavior cloning', 'imitation learning',
    'inverse reinforcement learning', 'irl', 'apprenticeship learning', 'preference learning',
    'reward learning', 'reward modeling', 'human feedback', 'rlhf', 'constitutional ai',
    'safe reinforcement learning', 'constrained rl', 'risk-sensitive rl', 'robust rl',
    'distributional rl', 'risk measures', 'cvar', 'var', 'spectral risk measures',
    'exploration', 'exploitation', 'exploration-exploitation tradeoff', 'multi-armed bandits',
    'contextual bandits', 'linear bandits', 'combinatorial bandits', 'dueling bandits',
    'pure exploration', 'best arm identification', 'fixed budget', 'fixed confidence',
    'regret minimization', 'cumulative regret', 'simple regret', 'minimax regret',
    'bayesian regret', 'frequentist regret', 'high-probability bounds', 'pac bounds',
    'sample complexity', 'generalization', 'finite-sample analysis', 'asymptotic analysis',
    
    # Time Series & Forecasting - Extended
    'time series', 'forecasting', 'time series analysis', 'temporal modeling', 'sequence modeling',
    'anomaly detection', 'outlier detection', 'novelty detection', 'change point detection',
    'trend analysis', 'seasonal decomposition', 'cyclical patterns', 'irregular components',
    'additive decomposition', 'multiplicative decomposition', 'x-11', 'x-12-arima', 'x-13arima-seats',
    'seats', 'tramo-seats', 'seasonal adjustment', 'calendar effects', 'trading day effects',
    'easter effects', 'holiday effects', 'intervention analysis', 'outlier detection',
    'level shifts', 'temporary changes', 'additive outliers', 'innovative outliers',
    'structural breaks', 'regime switching', 'markov switching', 'threshold models',
    'smooth transition', 'exponential smoothing', 'holt-winters', 'double exponential smoothing',
    'triple exponential smoothing', 'state space models', 'kalman filter', 'particle filter',
    'unscented kalman filter', 'extended kalman filter', 'ensemble kalman filter',
    'box-jenkins methodology', 'arima', 'autoregressive', 'ar', 'moving average', 'ma',
    'autoregressive moving average', 'arma', 'seasonal arima', 'sarima', 'fractional arima',
    'farima', 'vector autoregression', 'var', 'vector error correction', 'vec',
    'cointegration', 'error correction model', 'ecm', 'johansen test', 'engle-granger test',
    'unit root tests', 'adf test', 'augmented dickey-fuller', 'pp test', 'phillips-perron',
    'kpss test', 'stationarity', 'non-stationarity', 'differencing', 'integration',
    'spurious regression', 'granger causality', 'impulse response', 'variance decomposition',
    'forecast evaluation', 'forecast accuracy', 'forecast combination', 'forecast encompassing',
    'backtesting', 'walk-forward analysis', 'expanding window', 'rolling window',
    'cross-validation', 'time series cross-validation', 'blocked cross-validation',
    'hierarchical forecasting', 'forecast reconciliation', 'bottom-up', 'top-down',
    'middle-out', 'optimal reconciliation', 'mint', 'coherent forecasts',
    
    # Robotics & Automation - Extended
    'robotics', 'autonomous systems', 'robot navigation', 'path planning', 'motion planning',
    'robotic manipulation', 'human-robot interaction', 'hri', 'swarm robotics', 'mobile robotics',
    'industrial robotics', 'service robotics', 'medical robotics', 'surgical robotics',
    'rehabilitation robotics', 'assistive robotics', 'social robotics', 'companion robots',
    'educational robotics', 'entertainment robotics', 'military robotics', 'defense robotics',
    'search and rescue robotics', 'disaster response', 'space robotics', 'underwater robotics',
    'aerial robotics', 'drone technology', 'unmanned aerial vehicles', 'uav', 'quadcopters',
    'fixed-wing', 'vtol', 'autonomous vehicles', 'self-driving cars', 'adas',
    'advanced driver assistance systems', 'collision avoidance', 'lane keeping', 'adaptive cruise control',
    'automatic emergency braking', 'blind spot monitoring', 'parking assistance', 'valet parking',
    'vehicle-to-vehicle communication', 'v2v', 'vehicle-to-infrastructure', 'v2i',
    'vehicle-to-everything', 'v2x', 'connected vehicles', 'intelligent transportation systems',
    'its', 'traffic management', 'smart cities', 'urban planning', 'mobility as a service',
    'maas', 'shared mobility', 'ride sharing', 'car sharing', 'micromobility',
    'electric vehicles', 'ev', 'battery management', 'charging infrastructure', 'range anxiety',
    'energy efficiency', 'regenerative braking', 'hybrid vehicles', 'fuel cells',
    'hydrogen vehicles', 'alternative fuels', 'biofuels', 'synthetic fuels', 'e-fuels',
    'kinematics', 'inverse kinematics', 'forward kinematics', 'jacobian', 'singularities',
    'workspace analysis', 'reachability', 'dexterity', 'manipulability', 'condition number',
    'dynamics', 'inverse dynamics', 'forward dynamics', 'lagrangian mechanics', 'newton-euler',
    'recursive newton-euler', 'rne', 'featherstone algorithm', 'articulated body algorithm',
    'contact dynamics', 'collision detection', 'collision response', 'friction models',
    'coulomb friction', 'viscous friction', 'joint friction', 'backlash', 'flexibility',
    'control systems', 'feedback control', 'feedforward control', 'pid control',
    'proportional-integral-derivative', 'lqr', 'linear quadratic regulator', 'lqg',
    'linear quadratic gaussian', 'h-infinity control', 'robust control', 'adaptive control',
    'sliding mode control', 'backstepping', 'lyapunov stability', 'passivity', 'impedance control',
    'force control', 'hybrid force-position control', 'compliance control', 'stiffness control',
    
    # Graph & Network Analysis - Extended
    'graph neural networks', 'gnn', 'social network analysis', 'network analysis', 'graph mining',
    'link prediction', 'node classification', 'graph classification', 'community detection',
    'graph embedding', 'network embedding', 'node embedding', 'edge embedding', 'subgraph embedding',
    'knowledge graphs', 'knowledge graph completion', 'knowledge graph embedding', 'semantic networks',
    'ontologies', 'taxonomies', 'concept hierarchies', 'entity resolution', 'entity linking',
    'relation extraction', 'triple extraction', 'fact extraction', 'open information extraction',
    'citation networks', 'collaboration networks', 'co-authorship networks', 'bibliometrics',
    'scientometrics', 'altmetrics', 'impact factor', 'h-index', 'citation analysis',
    'network motifs', 'graphlets', 'subgraph counting', 'subgraph matching', 'graph isomorphism',
    'graph similarity', 'graph distance', 'graph kernels', 'shortest path', 'all-pairs shortest path',
    'single-source shortest path', 'dijkstra algorithm', 'bellman-ford algorithm', 'floyd-warshall',
    'betweenness centrality', 'closeness centrality', 'eigenvector centrality', 'pagerank',
    'hits algorithm', 'hubs and authorities', 'katz centrality', 'bonacich centrality',
    'degree centrality', 'clustering coefficient', 'transitivity', 'small world networks',
    'scale-free networks', 'random graphs', 'erdos-renyi', 'barabasi-albert', 'watts-strogatz',
    'configuration model', 'stochastic block model', 'sbm', 'degree-corrected sbm',
    'mixed membership', 'overlapping communities', 'hierarchical communities', 'temporal networks',
    'dynamic networks', 'evolving networks', 'network evolution', 'link formation', 'link dissolution',
    'network resilience', 'robustness', 'vulnerability', 'cascading failures', 'percolation',
    'epidemic spreading', 'information diffusion', 'influence maximization', 'viral marketing',
    'social influence', 'homophily', 'preferential attachment', 'triadic closure',
    'structural balance', 'signed networks', 'positive edges', 'negative edges', 'polarization',
    'echo chambers', 'filter bubbles', 'opinion dynamics', 'voter model', 'threshold models',
    
    # Audio & Speech Processing - Extended
    'speech recognition', 'automatic speech recognition', 'asr', 'speech synthesis', 'speech generation',
    'text-to-speech', 'tts', 'voice conversion', 'voice cloning', 'speaker recognition',
    'speaker verification', 'speaker identification', 'speaker diarization', 'voice activity detection',
    'vad', 'speech enhancement', 'noise reduction', 'noise suppression', 'denoising',
    'echo cancellation', 'acoustic echo cancellation', 'aec', 'beamforming', 'microphone arrays',
    'source separation', 'blind source separation', 'bss', 'independent component analysis', 'ica',
    'non-negative matrix factorization', 'nmf', 'cocktail party problem', 'multi-speaker',
    'speech separation', 'monaural separation', 'binaural separation', 'spatial audio',
    'audio processing', 'digital signal processing', 'dsp', 'fourier transform', 'fft',
    'short-time fourier transform', 'stft', 'spectrogram', 'mel-spectrogram', 'mfcc',
    'mel-frequency cepstral coefficients', 'plp', 'perceptual linear prediction', 'lpc',
    'linear predictive coding', 'formants', 'fundamental frequency', 'f0', 'pitch',
    'pitch tracking', 'pitch estimation', 'voicing detection', 'voiced', 'unvoiced',
    'phoneme recognition', 'phoneme classification', 'acoustic modeling', 'language modeling',
    'pronunciation modeling', 'lexicon', 'phonetic alphabet', 'ipa', 'arpabet', 'sampa',
    'prosody', 'prosodic features', 'intonation', 'stress', 'rhythm', 'accent', 'tone',
    'tonal languages', 'pitch accent', 'word stress', 'sentence stress', 'focus',
    'boundary detection', 'phrase boundaries', 'pause detection', 'breath detection',
    'emotion recognition', 'emotional speech', 'affective computing', 'sentiment analysis',
    'paralinguistics', 'computational paralinguistics', 'speaker state', 'speaker traits',
    'age estimation', 'gender recognition', 'accent recognition', 'language identification',
    'dialect recognition', 'pathological speech', 'dysarthria', 'apraxia', 'stuttering',
    'speech disorders', 'speech therapy', 'pronunciation training', 'computer-assisted pronunciation',
    'second language acquisition', 'l2 speech', 'foreign accent', 'intelligibility',
    'music information retrieval', 'mir', 'music analysis', 'music generation', 'music synthesis',
    'chord recognition', 'key detection', 'tempo estimation', 'beat tracking', 'onset detection',
    'music structure analysis', 'music similarity', 'music recommendation', 'playlist generation',
    'genre classification', 'instrument recognition', 'singing voice', 'melody extraction',
    'harmony analysis', 'music theory', 'computational musicology', 'audio fingerprinting',
    'audio matching', 'content-based audio retrieval', 'query by humming', 'audio thumbnailing',
    
    # Finance & Economics - Extended
    'finance', 'financial modeling', 'quantitative finance', 'algorithmic trading', 'high-frequency trading',
    'hft', 'market making', 'liquidity provision', 'order book dynamics', 'market microstructure',
    'price discovery', 'market efficiency', 'efficient market hypothesis', 'behavioral finance',
    'sentiment analysis', 'news analytics', 'social media sentiment', 'alternative data',
    'satellite data', 'credit scoring', 'credit risk', 'default prediction', 'bankruptcy prediction',
    'fraud detection', 'anti-money laundering', 'aml', 'know your customer', 'kyc',
    'regulatory compliance', 'basel ii', 'basel iii', 'solvency ii', 'mifid', 'dodd-frank',
    'risk management', 'market risk', 'credit risk', 'operational risk', 'liquidity risk',
    'value at risk', 'var', 'conditional value at risk', 'cvar', 'expected shortfall',
    'stress testing', 'scenario analysis', 'monte carlo simulation', 'historical simulation',
    'parametric methods', 'non-parametric methods', 'extreme value theory', 'evt',
    'portfolio optimization', 'mean-variance optimization', 'black-litterman', 'risk parity',
    'factor investing', 'smart beta', 'alternative risk premia', 'momentum', 'value',
    'quality', 'low volatility', 'size', 'profitability', 'investment', 'carry',
    'mean reversion', 'pairs trading', 'statistical arbitrage', 'market neutral',
    'long-short equity', 'hedge funds', 'mutual funds', 'etfs', 'pension funds',
    'insurance', 'actuarial science', 'life insurance', 'property casualty', 'reinsurance',
    'catastrophe modeling', 'natural disasters', 'climate risk', 'esg investing',
    'sustainable finance', 'green bonds', 'carbon credits', 'impact investing',
    'cryptocurrency', 'blockchain', 'bitcoin', 'ethereum', 'defi', 'decentralized finance',
    'smart contracts', 'tokenization', 'nfts', 'stablecoins', 'central bank digital currencies',
    'cbdc', 'payment systems', 'fintech', 'regtech', 'insurtech', 'wealthtech',
    'robo-advisors', 'digital banking', 'open banking', 'api banking', 'embedded finance',
    
    # Bioinformatics & Genomics - Extended
    'bioinformatics', 'computational biology', 'genomics', 'proteomics', 'metabolomics',
    'transcriptomics', 'epigenomics', 'metagenomics', 'phylogenomics', 'comparative genomics',
    'structural genomics', 'functional genomics', 'systems biology', 'network biology',
    'gene expression', 'rna-seq', 'single-cell rna-seq', 'scrna-seq', 'spatial transcriptomics',
    'chip-seq', 'atac-seq', 'cut&run', 'cut&tag', 'hi-c', 'chromosome conformation',
    '4c-seq', '5c-seq', 'capture-c', 'chromatin', 'histone modifications', 'dna methylation',
    'bisulfite sequencing', 'wgbs', 'rrbs', 'hydroxymethylation', 'chromatin accessibility',
    'transcription factors', 'tf binding', 'gene regulation', 'regulatory networks',
    'gene regulatory networks', 'grn', 'transcriptional networks', 'protein-protein interactions',
    'ppi', 'interactome', 'pathway analysis', 'gene ontology', 'go', 'kegg', 'reactome',
    'wikipathways', 'pathway enrichment', 'functional enrichment', 'gsea', 'gene set enrichment',
    'differential expression', 'deseq2', 'edger', 'limma', 'fold change', 'p-value',
    'false discovery rate', 'fdr', 'multiple testing correction', 'bonferroni', 'benjamini-hochberg',
    'sequence alignment', 'blast', 'psi-blast', 'hmmer', 'pfam', 'protein domains',
    'protein families', 'protein structure', 'structure prediction', 'alphafold', 'colabfold',
    'molecular dynamics', 'md', 'protein folding', 'protein stability', 'thermodynamics',
    'drug discovery', 'drug design', 'virtual screening', 'molecular docking', 'qsar',
    'pharmacokinetics', 'pharmacodynamics', 'admet', 'toxicity prediction', 'side effects',
    'drug-target interactions', 'polypharmacology', 'drug repurposing', 'drug resistance',
    'precision medicine', 'personalized medicine', 'pharmacogenomics', 'biomarkers',
    'companion diagnostics', 'theranostics', 'liquid biopsy', 'circulating tumor cells',
    'ctc', 'circulating tumor dna', 'ctdna', 'cell-free dna', 'cfdna', 'extracellular vesicles',
    'exosomes', 'mirna', 'lncrna', 'non-coding rna', 'regulatory rna', 'ribosomal rna',
    'transfer rna', 'snorna', 'pirna', 'sirna', 'shrna', 'crispr', 'cas9', 'cas12',
    'gene editing', 'genome editing', 'base editing', 'prime editing', 'epigenome editing',
    
    # Materials Science & Chemistry - Extended
    'materials science', 'materials informatics', 'computational materials science', 'density functional theory',
    'dft', 'molecular dynamics', 'monte carlo methods', 'phase field modeling', 'finite element',
    'crystal structure prediction', 'materials discovery', 'high-throughput screening', 'combinatorial',
    'materials databases', 'materials project', 'nomad', 'aflow', 'oqmd', 'crystallography',
    'x-ray diffraction', 'xrd', 'electron microscopy', 'sem', 'tem', 'afm', 'stm',
    'spectroscopy', 'infrared', 'raman', 'nmr', 'esr', 'xps', 'auger', 'mass spectrometry',
    'chromatography', 'hplc', 'gc-ms', 'lc-ms', 'analytical chemistry', 'separation science',
    'chemical analysis', 'quantitative analysis', 'qualitative analysis', 'trace analysis',
    'environmental analysis', 'food analysis', 'pharmaceutical analysis', 'forensic chemistry',
    'green chemistry', 'sustainable chemistry', 'atom economy', 'catalysis', 'biocatalysis',
    'enzyme engineering', 'directed evolution', 'protein engineering', 'metabolic engineering',
    'synthetic biology', 'systems biology', 'chemical biology', 'bioorganic chemistry',
    'medicinal chemistry', 'drug discovery', 'lead optimization', 'structure-activity relationships',
    'sar', 'quantitative structure-activity relationships', 'qsar', 'pharmacophore',
    'molecular recognition', 'host-guest chemistry', 'supramolecular chemistry', 'self-assembly',
    'nanomaterials', 'nanoparticles', 'quantum dots', 'carbon nanotubes', 'graphene',
    '2d materials', 'mxenes', 'transition metal dichalcogenides', 'tmdc', 'perovskites',
    'metal-organic frameworks', 'mof', 'covalent organic frameworks', 'cof', 'porous materials',
    'zeolites', 'activated carbon', 'aerogels', 'hydrogels', 'polymers', 'plastics',
    'elastomers', 'thermosets', 'thermoplastics', 'biodegradable polymers', 'bioplastics',
    'smart materials', 'shape memory alloys', 'piezoelectric materials', 'ferroelectric',
    'magnetostrictive', 'photonic materials', 'metamaterials', 'phononic crystals',
    'energy materials', 'battery materials', 'fuel cells', 'solar cells', 'photovoltaics',
    'thermoelectrics', 'superconductors', 'magnetic materials', 'electronic materials',
    'semiconductors', 'dielectrics', 'insulators', 'conductors', 'optical materials',
    
    # Climate Science & Environmental Monitoring - Extended
    'climate science', 'climate modeling', 'global warming', 'climate change', 'greenhouse gases',
    'carbon dioxide', 'co2', 'methane', 'ch4', 'nitrous oxide', 'n2o', 'fluorinated gases',
    'carbon cycle', 'nitrogen cycle', 'water cycle', 'hydrological cycle', 'precipitation',
    'evaporation', 'transpiration', 'evapotranspiration', 'runoff', 'groundwater',
    'surface water', 'atmospheric circulation', 'ocean circulation', 'thermohaline circulation',
    'gulf stream', 'el nino', 'la nina', 'enso', 'north atlantic oscillation', 'nao',
    'arctic oscillation', 'ao', 'pacific decadal oscillation', 'pdo', 'atlantic multidecadal oscillation',
    'amo', 'madden-julian oscillation', 'mjo', 'monsoons', 'tropical cyclones', 'hurricanes',
    'typhoons', 'extreme weather', 'heat waves', 'cold waves', 'droughts', 'floods',
    'wildfires', 'sea level rise', 'ice sheets', 'glaciers', 'permafrost', 'albedo',
    'radiative forcing', 'feedback loops', 'tipping points', 'irreversible changes',
    'paleoclimate', 'ice cores', 'tree rings', 'coral reefs', 'sediment cores', 'fossils',
    'proxy data', 'reconstruction', 'holocene', 'pleistocene', 'last glacial maximum',
    'little ice age', 'medieval warm period', 'younger dryas', 'anthropocene',
    'environmental monitoring', 'air quality', 'water quality', 'soil quality', 'pollution',
    'atmospheric pollution', 'particulate matter', 'pm2.5', 'pm10', 'ozone', 'nitrogen oxides',
    'nox', 'sulfur dioxide', 'so2', 'carbon monoxide', 'co', 'volatile organic compounds',
    'voc', 'polycyclic aromatic hydrocarbons', 'pah', 'heavy metals', 'pesticides',
    'persistent organic pollutants', 'pop', 'endocrine disruptors', 'microplastics',
    'nanoplastics', 'marine pollution', 'ocean acidification', 'eutrophication', 'algal blooms',
    'dead zones', 'biodiversity', 'ecosystem services', 'habitat loss', 'deforestation',
    'desertification', 'land use change', 'urbanization', 'agriculture', 'sustainability',
    'renewable energy', 'solar energy', 'wind energy', 'hydroelectric', 'geothermal',
    'biomass', 'biofuels', 'energy efficiency', 'energy storage', 'smart grids',
    'carbon capture', 'carbon storage', 'ccs', 'direct air capture', 'dac',
    
    # Autonomous Systems & Control - Extended
    'autonomous systems', 'control theory', 'control systems', 'feedback control', 'feedforward control',
    'linear control', 'nonlinear control', 'adaptive control', 'robust control', 'optimal control',
    'model predictive control', 'mpc', 'pid control', 'fuzzy control', 'neural control',
    'sliding mode control', 'backstepping', 'lyapunov stability', 'passivity', 'dissipativity',
    'h-infinity control', 'h2 control', 'linear quadratic regulator', 'lqr', 'linear quadratic gaussian',
    'lqg', 'kalman filter', 'extended kalman filter', 'unscented kalman filter', 'particle filter',
    'observer design', 'state estimation', 'parameter estimation', 'system identification',
    'grey box modeling', 'black box modeling', 'white box modeling', 'transfer functions',
    'state space models', 'controllability', 'observability', 'stability', 'stabilizability',
    'detectability', 'reachability', 'minimum phase', 'non-minimum phase', 'zeros', 'poles',
    'frequency response', 'bode plots', 'nyquist plots', 'root locus', 'nichols charts',
    'gain margin', 'phase margin', 'bandwidth', 'settling time', 'overshoot', 'rise time',
    'steady-state error', 'tracking', 'regulation', 'disturbance rejection', 'noise rejection',
    'sensitivity', 'complementary sensitivity', 'mixed sensitivity', 'structured singular value',
    'mu-synthesis', 'structured uncertainty', 'unstructured uncertainty', 'parametric uncertainty',
    'norm-bounded uncertainty', 'polytopic uncertainty', 'interval uncertainty', 'stochastic uncertainty',
    'distributed control', 'decentralized control', 'networked control', 'wireless control',
    'event-triggered control', 'self-triggered control', 'quantized control', 'sampled-data control',
    'hybrid systems', 'switched systems', 'piecewise affine systems', 'complementarity systems',
    'discrete event systems', 'petri nets', 'automata', 'supervisory control', 'fault detection',
    'fault diagnosis', 'fault-tolerant control', 'reconfigurable control', 'graceful degradation',
    'multi-agent systems', 'consensus', 'synchronization', 'flocking', 'formation control',
    'cooperative control', 'distributed optimization', 'game-theoretic control', 'mechanism design',
    'auction mechanisms', 'market-based control', 'economic dispatch', 'optimal power flow',
    
    # Advanced Manufacturing & Industry 4.0 - Extended
    'industry 4.0', 'smart manufacturing', 'digital manufacturing', 'cyber-physical systems',
    'cps', 'industrial internet of things', 'iiot', 'digital twins', 'virtual commissioning',
    'simulation-driven design', 'model-based systems engineering', 'mbse', 'systems engineering',
    'product lifecycle management', 'plm', 'manufacturing execution systems', 'mes',
    'enterprise resource planning', 'erp', 'supply chain management', 'scm', 'logistics',
    'inventory management', 'demand forecasting', 'production planning', 'scheduling',
    'lean manufacturing', 'just-in-time', 'jit', 'total quality management', 'tqm',
    'six sigma', 'statistical process control', 'spc', 'design of experiments', 'doe',
    'quality control', 'quality assurance', 'inspection', 'metrology', 'measurement',
    'coordinate measuring machines', 'cmm', 'optical measurement', 'laser scanning',
    'structured light scanning', 'photogrammetry', 'reverse engineering', 'rapid prototyping',
    'additive manufacturing', '3d printing', 'stereolithography', 'sla', 'selective laser sintering',
    'sls', 'fused deposition modeling', 'fdm', 'electron beam melting', 'ebm',
    'direct metal laser sintering', 'dmls', 'binder jetting', 'material jetting',
    'sheet lamination', 'directed energy deposition', 'ded', 'wire arc additive manufacturing',
    'waam', 'hybrid manufacturing', 'subtractive manufacturing', 'machining', 'milling',
    'turning', 'drilling', 'grinding', 'polishing', 'surface finishing', 'coating',
    'plating', 'anodizing', 'heat treatment', 'welding', 'brazing', 'soldering',
    'assembly', 'robotics', 'industrial robots', 'collaborative robots', 'cobots',
    'robot programming', 'path planning', 'motion control', 'force control', 'vision systems',
    'machine vision', 'image processing', 'pattern recognition', 'defect detection',
    'surface inspection', 'dimensional measurement', 'gauging', 'sensors', 'actuators',
    'plc', 'programmable logic controllers', 'scada', 'hmi', 'human-machine interface',
    'automation', 'process automation', 'factory automation', 'building automation',
    'home automation', 'smart homes', 'smart cities', 'smart grid', 'energy management',
    
    # Quantum Computing & Physics - Extended
    'quantum computing', 'quantum algorithms', 'quantum machine learning', 'quantum supremacy',
    'quantum advantage', 'quantum error correction', 'quantum cryptography', 'quantum communication',
    'quantum key distribution', 'qkd', 'quantum internet', 'quantum networks', 'quantum sensing',
    'quantum metrology', 'quantum simulation', 'quantum annealing', 'adiabatic quantum computing',
    'gate-based quantum computing', 'superconducting qubits', 'trapped ion qubits', 'photonic qubits',
    'topological qubits', 'neutral atom qubits', 'quantum dots', 'quantum wells', 'quantum wires',
    'spin qubits', 'charge qubits', 'flux qubits', 'transmon qubits', 'josephson junctions',
    'quantum gates', 'pauli gates', 'hadamard gate', 'cnot gate', 'toffoli gate',
    'controlled gates', 'rotation gates', 'phase gates', 'quantum circuits', 'quantum registers',
    'quantum entanglement', 'quantum superposition', 'quantum interference', 'quantum decoherence',
    'quantum noise', 'depolarizing noise', 'amplitude damping', 'phase damping', 'bit flip',
    'phase flip', 'quantum channels', 'quantum operations', 'quantum measurements', 'born rule',
    'quantum state tomography', 'process tomography', 'quantum characterization', 'benchmarking',
    'randomized benchmarking', 'quantum volume', 'cross-entropy benchmarking', 'quantum fidelity',
    'trace distance', 'diamond distance', 'quantum information theory', 'quantum entropy',
    'von neumann entropy', 'quantum mutual information', 'quantum discord', 'quantum coherence',
    'quantum correlations', 'bell inequalities', 'chsh inequality', 'local hidden variables',
    'contextuality', 'kochen-specker theorem', 'gleason theorem', 'quantum foundations',
    'interpretations of quantum mechanics', 'copenhagen interpretation', 'many-worlds interpretation',
    'pilot-wave theory', 'objective collapse theories', 'qbism', 'relational quantum mechanics'
]

METHODS = [
    # Neural Network Architectures
    'neural networks', 'artificial neural networks', 'deep neural networks', 'multilayer perceptron',
    'mlp', 'feedforward networks', 'backpropagation', 'gradient descent', 'stochastic gradient descent',
    'sgd', 'adam', 'rmsprop', 'adagrad', 'momentum', 'batch normalization', 'layer normalization',
    'dropout', 'early stopping', 'regularization', 'l1 regularization', 'l2 regularization',
    'residual connections', 'skip connections', 'highway networks', 'siren networks',
    
    # Convolutional Networks
    'cnn', 'convolutional neural networks', 'convolution', 'pooling', 'max pooling', 'average pooling',
    'residual networks', 'resnet', 'densenet', 'inception', 'mobilenet', 'efficientnet',
    'vgg', 'alexnet', 'googlenet', 'squeezenet', 'shufflenet', 'nasnet', 'neural architecture search',
    'depthwise separable convolution', 'convnext', 'regnet', 'vision transformer', 'vit',
    
    # Recurrent Networks
    'rnn', 'recurrent neural networks', 'lstm', 'long short-term memory', 'gru', 'gated recurrent unit',
    'bidirectional lstm', 'sequence-to-sequence', 'seq2seq', 'encoder-decoder', 'attention mechanism',
    'temporal convolutional networks', 'tcn', 'echo state networks', 'elman network', 'jordan network',
    
    # Transformer Architecture
    'transformer', 'attention', 'self-attention', 'multi-head attention', 'positional encoding',
    'bert', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 't5', 'bart', 'roberta', 'electra', 'deberta',
    'xlnet', 'albert', 'distilbert', 'longformer', 'bigbird', 'performer', 'linformer',
    'switch transformer', 'palm', 'lamda', 'chatgpt', 'claude', 'bloom', 'opt', 'flan-t5',
    'pegasus', 'marian', 'm2m100', 'mt5', 'prophetnet', 'reformer', 'funnel transformer', 'swin transformer',
    
    # Generative Models
    'generative adversarial networks', 'gan', 'dcgan', 'wgan', 'stylegan', 'cyclegan', 'pix2pix',
    'variational autoencoder', 'vae', 'autoencoder', 'denoising autoencoder', 'sparse autoencoder',
    'diffusion models', 'stable diffusion', 'dalle', 'midjourney', 'imagen', 'parti',
    'normalizing flows', 'real nvp', 'glow', 'masked autoregressive flows', 'score-based generative models',
    
    # Classical Machine Learning
    'support vector machines', 'svm', 'kernel methods', 'gaussian processes', 'bayesian methods',
    'naive bayes', 'logistic regression', 'linear regression', 'polynomial regression',
    'ridge regression', 'lasso regression', 'elastic net', 'principal component analysis', 'pca',
    'independent component analysis', 'ica', 'linear discriminant analysis', 'lda',
    'quadratic discriminant analysis', 'qda', 'perceptron', 'nearest neighbors', 'k-nearest neighbors', 'knn',
    
    # Tree-based Methods
    'decision trees', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost',
    'adaboost', 'extra trees', 'isolation forest', 'bagging', 'bootstrap aggregating', 'oblique decision trees',
    
    # Clustering & Unsupervised
    'k-means', 'k-means++', 'hierarchical clustering', 'dbscan', 'gaussian mixture models', 'gmm',
    'expectation maximization', 'em algorithm', 'spectral clustering', 'mean shift',
    'affinity propagation', 'birch', 'optics', 't-sne', 'umap', 'isomap', 'locally linear embedding',
    'self-organizing maps', 'som', 'factor analysis', 'non-negative matrix factorization', 'nmf',
    
    # Reinforcement Learning
    'q-learning', 'deep q-networks', 'dqn', 'policy gradient', 'actor-critic', 'a3c', 'ppo',
    'proximal policy optimization', 'trpo', 'trust region policy optimization', 'ddpg',
    'deep deterministic policy gradient', 'sac', 'soft actor-critic', 'rainbow', 'dueling dqn',
    'prioritized experience replay', 'monte carlo tree search', 'mcts', 'alphago', 'alphazero',
    'continuous q-learning', 'multi-agent reinforcement learning', 'multi-agent rl', 'imitation learning',
    
    # Graph Methods
    'graph neural networks', 'gnn', 'graph convolutional networks', 'gcn', 'graphsage',
    'graph attention networks', 'gat', 'message passing', 'node2vec', 'deepwalk', 'metapath2vec',
    'graph embedding', 'network embedding', 'community detection algorithms', 'pagerank', 'link prediction',
    'graph autoencoder', 'graph transformers', 'heterogeneous graph neural networks', 'hgnn',
    
    # Optimization Algorithms
    'evolutionary algorithms', 'genetic algorithms', 'particle swarm optimization', 'pso',
    'ant colony optimization', 'simulated annealing', 'tabu search', 'hill climbing',
    'differential evolution', 'cma-es', 'hyperparameter optimization', 'bayesian optimization',
    'grid search', 'random search', 'optuna', 'hyperopt', 'neural architecture search', 'nas',
    'convex optimization', 'non-convex optimization', 'gradient-free optimization', 'metaheuristics',
    
    # Ensemble Methods
    'ensemble learning', 'voting classifier', 'stacking', 'blending', 'model averaging',
    'bayesian model averaging', 'mixture of experts', 'boosting', 'bagging', 'cross-validation',
    'bootstrap aggregating', 'weighted ensemble', 'snapshot ensemble',
    
    # Advanced Techniques
    'knowledge distillation', 'model compression', 'pruning', 'quantization', 'fine-tuning',
    'domain adaptation', 'transfer learning', 'multi-task learning', 'meta-learning',
    'few-shot learning', 'zero-shot learning', 'prompt engineering', 'in-context learning',
    'chain-of-thought', 'retrieval-augmented generation', 'rag', 'tool use', 'function calling',
    'contrastive learning', 'self-supervised learning', 'representation learning', 'curriculum learning',
    'adversarial training', 'robust training', 'causal inference', 'causal discovery', 'explainable AI', 'xai'
]


DATASETS = [
    # Computer Vision Datasets
    'imagenet', 'cifar-10', 'cifar-100', 'mnist', 'fashion-mnist', 'emnist', 'svhn',
    'coco', 'microsoft coco', 'pascal voc', 'open images', 'ava', 'kinetics', 'youtube-8m',
    'places365', 'sun397', 'caltech-101', 'caltech-256', 'stanford cars', 'fgvc-aircraft',
    'food-101', 'pets', 'flowers-102', 'dtd', 'eurosat', 'resisc45', 'ucf101', 'hmdb51',
    'something-something', 'moments in time', 'activitynet', 'charades', 'epic-kitchens',
    'celeba', 'lfw', 'vggface', 'wider face', 'aflw', 'helen', '300w', 'wflw',
    'cityscapes', 'ade20k', 'pascal context', 'nyuv2', 'sun rgbd', 'scannet',
    'kitti', 'nuscenes', 'waymo', 'berkeley deepdrive', 'apolloscape', 'comma2k19',
    
    # Medical Imaging Datasets
    'mimic', 'mimic-cxr', 'chexpert', 'nih chestxray', 'padchest', 'vinbigdata',
    'rsna pneumonia', 'siim-acr', 'stoic2021', 'covid-ct', 'covidx', 'brats', 'lidc-idri',
    'luna16', 'isic', 'ham10000', 'ph2', 'dermnet', 'fitzpatrick17k', 'papsmear',
    'physionet', 'ptb-xl', 'chapman', 'georgia', 'cpsc2018', 'cinc2020', 'mit-bih',
    'apnea-ecg', 'bidmc', 'fantasia', 'sleep-edf', 'ddsm', 'cbis-ddsm', 'inbreast',
    
    # Natural Language Processing Datasets
    'glue', 'superglue', 'squad', 'squad 2.0', 'natural questions', 'ms marco', 'quac',
    'coqa', 'hotpotqa', 'triviaqa', 'searchqa', 'narrativeqa', 'race', 'arc',
    'hellaswag', 'winogrande', 'piqa', 'siqa', 'commonsenqa', 'copa', 'wsc', 'rte',
    'wnli', 'sst', 'cola', 'qqp', 'qnli', 'sts-b', 'mnli', 'snli', 'xnli', 'anli',
    'imdb', 'yelp reviews', 'amazon reviews', 'rotten tomatoes', 'stanford sentiment',
    'conll-2003', 'ontonotes', 'wikiner', 'few-nerd', 'tacred', 'semeval', 're-tacred',
    'wmt', 'opus', 'multi30k', 'iwslt', 'flores', 'ted talks', 'opensubtitles',
    'common crawl', 'the pile', 'c4', 'openwebtext', 'bookcorpus', 'wikipedia',
    'gutenberg', 'stories', 'realnews', 'webtext', 'cc-news', 'cc-stories',
    
    # Speech and Audio Datasets
    'librispeech', 'common voice', 'voxceleb', 'switchboard', 'fisher', 'callhome',
    'tedlium', 'ami', 'icsi', 'chime', 'reverb', 'dcase', 'esc-50', 'urbansound8k',
    'audioset', 'fma', 'gtzan', 'million song dataset', 'musicnet', 'nsynth',
    'ljspeech', 'blizzard', 'cmu arctic', 'vctk', 'hi-fi', 'dns challenge',
    
    # Graph and Network Datasets
    'cora', 'citeseer', 'pubmed', 'reddit', 'ppi', 'amazon', 'yelp', 'flickr',
    'facebook', 'twitter', 'blog catalog', 'wikipedia network', 'dblp', 'arxiv',
    'ogb', 'tu datasets', 'snap datasets', 'networkx datasets', 'graph-saint',
    
    # Time Series Datasets
    'ucr time series', 'uea multivariate', 'electricity', 'traffic', 'exchange rate',
    'solar energy', 'air quality', 'nasdaq', 'sp500', 'forex', 'bitcoin',
    'energy consumption', 'household power', 'appliances energy', 'eeg',
    'ecg', 'emg', 'accelerometer', 'gyroscope', 'sensor data',
    
    # Reinforcement Learning Environments
    'atari', 'openai gym', 'mujoco', 'roboschool', 'pybullet', 'deepmind lab',
    'starcraft ii', 'dota 2', 'minecraft', 'procgen', 'cartpole', 'mountaincar',
    'pendulum', 'lunar lander', 'bipedal walker', 'ant', 'halfcheetah', 'hopper',
    'humanoid', 'walker2d', 'reacher', 'pusher', 'striker', 'thrower',
    
    # Specialized Domain Datasets
    'movielens', 'netflix prize', 'amazon product', 'goodreads', 'lastfm',
    'booking.com', 'trivago', 'expedia', 'airbnb', 'uber', 'lyft', 'taxi',
    'creditcard fraud', 'kdd cup', 'adult census', 'bank marketing', 'wine quality',
    'breast cancer', 'diabetes', 'heart disease', 'titanic', 'boston housing',
    'california housing', 'diamonds', 'tips', 'flights', 'mpg', 'mtcars',
    
    # Recent Large-Scale Datasets
    'laion', 'conceptual captions', 'yfcc100m', 'redcaps', 'wit', 'localized narratives',
    'cc12m', 'cc3m', 'sbu captions', 'visual genome', 'nocaps', 'flickr30k',
    'mscoco captions', 'textcaps', 'vizwiz', 'gqa', 'clevr', 'nlvr', 'nlvr2',
    'webqa', 'okvqa', 'a-okvqa', 'scienceqa', 'ai2d', 'tqa', 'dvqa', 'figureqa'
]

METRICS = [
    # Classification Metrics
    'accuracy', 'precision', 'recall', 'f1-score', 'f1 score', 'f-measure', 'f-beta',
    'sensitivity', 'specificity', 'true positive rate', 'false positive rate',
    'true negative rate', 'false negative rate', 'balanced accuracy', 'top-k accuracy',
    'auc', 'roc-auc', 'area under curve', 'roc curve', 'pr-auc', 'precision-recall curve',
    'average precision', 'map', 'mean average precision', 'ndcg', 'normalized dcg',
    'matthews correlation coefficient', 'mcc', 'cohen kappa', 'kappa score',
    'log loss', 'cross entropy', 'binary crossentropy', 'categorical crossentropy',
    'focal loss', 'dice coefficient', 'jaccard index', 'iou', 'intersection over union',
    
    # Regression Metrics
    'mean squared error', 'mse', 'root mean squared error', 'rmse', 'mean absolute error',
    'mae', 'mean absolute percentage error', 'mape', 'symmetric mape', 'smape',
    'r-squared', 'r2 score', 'adjusted r-squared', 'coefficient of determination',
    'mean squared logarithmic error', 'msle', 'huber loss', 'quantile loss',
    'explained variance', 'max error', 'median absolute error', 'medae',
    
    # Natural Language Processing Metrics
    'bleu', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'sacrebleu', 'chrf', 'chrf++',
    'rouge', 'rouge-1', 'rouge-2', 'rouge-l', 'rouge-w', 'rouge-s', 'rouge-su',
    'meteor', 'ter', 'translation error rate', 'bertscore', 'bleurt', 'comet',
    'cider', 'spice', 'wmd', 'word mover distance', 'sentence-bleu', 'corpus-bleu',
    'perplexity', 'bits per character', 'bpc', 'cross-entropy loss', 'likelihood',
    'exact match', 'em', 'partial match', 'token-level f1', 'squad score',
    
    # Computer Vision Metrics
    'pixel accuracy', 'mean iou', 'miou', 'dice score', 'hausdorff distance',
    'peak signal-to-noise ratio', 'psnr', 'structural similarity', 'ssim',
    'multiscale ssim', 'ms-ssim', 'learned perceptual image patch similarity', 'lpips',
    'frechet inception distance', 'fid', 'inception score', 'is', 'kernel inception distance',
    'kid', 'wasserstein distance', 'earth mover distance', 'emd', 'chamfer distance',
    'average precision', 'ap', 'map50', 'map75', 'coco metrics', 'pascal metrics',
    'object keypoint similarity', 'oks', 'panoptic quality', 'pq', 'segmentation quality',
    'recognition quality', 'rq', 'detection quality', 'dq', 'mean pixel accuracy',
    'frequency weighted iou', 'boundary f1', 'optical flow error', 'epe', 'endpoint error',
    
    # Information Retrieval Metrics
    'precision at k', 'p@k', 'recall at k', 'r@k', 'mean reciprocal rank', 'mrr',
    'hit rate', 'coverage', 'diversity', 'novelty', 'serendipity', 'catalog coverage',
    'intra-list diversity', 'aggregate diversity', 'personalization', 'popularity bias',
    'long tail', 'gini coefficient', 'entropy', 'expected reciprocal rank', 'err',
    
    # Ranking Metrics
    'kendall tau', 'spearman correlation', 'pearson correlation', 'rank correlation',
    'concordance index', 'c-index', 'auc', 'gini coefficient', 'somers d',
    'goodman kruskal gamma', 'weighted tau', 'top-k ranking', 'learning to rank metrics',
    
    # Time Series Metrics
    'mean absolute scaled error', 'mase', 'mean directional accuracy', 'mda',
    'tracking signal', 'forecast bias', 'theil u statistic', 'autocorrelation',
    'partial autocorrelation', 'ljung-box test', 'adf test', 'kpss test',
    'seasonal decomposition', 'trend accuracy', 'seasonal accuracy', 'residual accuracy',
    
    # Clustering Metrics
    'silhouette score', 'silhouette coefficient', 'calinski harabasz', 'davies bouldin',
    'adjusted rand index', 'ari', 'normalized mutual information', 'nmi',
    'homogeneity', 'completeness', 'v-measure', 'fowlkes mallows', 'rand index',
    'mutual information', 'contingency matrix', 'purity', 'entropy', 'conductance',
    'modularity', 'normalized cut', 'ratio cut', 'within cluster sum of squares', 'wcss',
    'between cluster sum of squares', 'bcss', 'inertia', 'distortion',
    
    # Statistical Metrics
    'confidence interval', 'p-value', 'statistical significance', 'effect size',
    'cohen d', 'glass delta', 'hedge g', 'cramers v', 'phi coefficient',
    'chi-square', 'chi2', 'anova', 'f-statistic', 't-statistic', 'z-score',
    'kolmogorov-smirnov', 'ks test', 'mann-whitney u', 'wilcoxon', 'friedman test',
    
    # Model Performance Metrics
    'training time', 'inference time', 'model size', 'memory usage', 'flops',
    'floating point operations', 'parameters', 'trainable parameters', 'model complexity',
    'computational cost', 'energy consumption', 'carbon footprint', 'throughput',
    'latency', 'fps', 'frames per second', 'real-time factor', 'speed',
    'convergence rate', 'learning curve', 'validation curve', 'training loss',
    'validation loss', 'test loss', 'generalization gap', 'overfitting', 'underfitting',
    
    # Fairness and Bias Metrics
    'demographic parity', 'equalized odds', 'equality of opportunity', 'calibration',
    'fairness through unawareness', 'counterfactual fairness', 'individual fairness',
    'group fairness', 'disparate impact', 'statistical parity', 'treatment equality',
    'conditional statistical parity', 'predictive parity', 'predictive equality',
    
    # Robustness Metrics
    'adversarial accuracy', 'robust accuracy', 'certified accuracy', 'attack success rate',
    'perturbation budget', 'epsilon ball', 'l0 norm', 'l1 norm', 'l2 norm', 'linf norm',
    'gradient norm', 'lipschitz constant', 'local lipschitz', 'smoothness',
    'corruption error', 'distribution shift', 'domain gap', 'calibration error',
    'expected calibration error', 'ece', 'maximum calibration error', 'mce',
    'brier score', 'reliability diagram', 'prediction interval coverage'
]

# Charger le modèle d’embeddings (rapide et léger)
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pré-calculer les embeddings des listes
_EMBEDDINGS = {
    "domain": {kw: _embedding_model.encode(kw, convert_to_tensor=True) for kw in DOMAIN_KEYWORDS},
    "methods": {kw: _embedding_model.encode(kw, convert_to_tensor=True) for kw in METHODS},
    "datasets": {kw: _embedding_model.encode(kw, convert_to_tensor=True) for kw in DATASETS},
    "metrics": {kw: _embedding_model.encode(kw, convert_to_tensor=True) for kw in METRICS},
}


def _maybe_load_spacy(lang: str):
    if not has_spacy:
        return None
    try:
        if lang == 'fr':
            return spacy.load('fr_core_news_sm')
        return spacy.load('en_core_web_sm')
    except Exception:
        return None


def extract_entities(text: str, lang: str = 'en', top_k: int = 2) -> Dict[str, List[str]]:
    """Extrait des entités à partir d’un texte, avec embeddings + noun chunks spaCy."""
    lowered = (text or '').lower()
    entities: Dict[str, List[str]] = {
        'domain': [],
        'methods': [],
        'datasets': [],
        'metrics': [],
        'keywords': []
    }

    # Embedding du texte
    query_emb = _embedding_model.encode(text, convert_to_tensor=True)

    # Chercher les entités les plus proches par similarité cosinus
    for category, emb_dict in _EMBEDDINGS.items():
        scores = {kw: float(util.cos_sim(query_emb, v)) for kw, v in emb_dict.items()}
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for ent, score in top_matches:
            if score > 0.5:  # seuil à ajuster
                entities[category].append(ent)

    # spaCy → extraire des keywords additionnels (noun_chunks)
    if has_spacy:
        nlp = _maybe_load_spacy(lang)
        if nlp is not None:
            try:
                doc = nlp(text)
                for chunk in doc.noun_chunks:
                    k = chunk.text.strip().lower()
                    if k and k not in entities['keywords']:
                        entities['keywords'].append(k)
            except Exception:
                pass

    # Supprimer doublons
    for key in entities:
        seen = set()
        unique = []
        for item in entities[key]:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        entities[key] = unique

    return entities
