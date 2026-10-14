# S1-Arena: benchmark sources and environments

Every dataset and simulator that S1-Arena draws from, with its link and license. Requests are counted per source; a request can hold several typed questions about one input.

- **Non-agentic tasks** (the first three groups): 220 sources, 43,226 requests, 13 domains.
- **Agentic tasks, static**: 48 sources (8,973 requests): agent single steps, static game positions and static embodied decisions.
- **Agentic tasks, closed loop**: 138 environments.

## Non-agentic tasks

### Text and knowledge

#### Basic text classification (5,000 requests, 26 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| entity_relation | Few-NERD supervised, fine-grained entity types (Ding et al., 2021) | 167 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/DFKI-SLT/few-nerd) |
| entity_relation | FewRel 1.0, the 80 public Wikidata relations (Han et al., 2018) | 167 | MIT | [link](https://github.com/thunlp/FewRel) |
| entity_relation | SemEval-2010 Task 8, relations between nominals (Hendrickx et al., 2010) | 166 | CC BY 3.0 | [link](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) |
| intent | CLINC150, plus variant with out-of-scope queries (Larson et al., 2019) | 250 | CC BY 3.0 | [link](https://huggingface.co/datasets/clinc/clinc_oos) |
| intent | MASSIVE, en-US (FitzGerald et al., 2022) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/mteb/amazon_massive_intent) |
| language_style | CEFR-SP, public part (Arase et al., 2022) | 167 | CC BY-SA 3.0 (Wiki-Auto part) / CC BY-NC-SA 4.0 (SCoRE part) | [link](https://huggingface.co/datasets/edesaras/CEFR-Sentence-Level-Annotations) |
| language_style | Formality scores (Pavlick and Tetreault, 2016) | 167 | CC BY 3.0 | [link](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) |
| language_style | WiLI-2018 Wikipedia language identification (Thoma, 2018) | 166 | ODbL 1.0 | [link](https://huggingface.co/datasets/MartinThoma/wili_2018) |
| multilabel | GoEmotions (Demszky et al., 2020) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/google-research-datasets/go_emotions) |
| multilabel | SemEval-2018 Task 1 Affect in Tweets, emotion classification E-c (Mohammad et al., 2018) | 250 | Research use only; do not redistribute (NRC) (rebuilt by a loader) | [link](https://saifmohammad.com/WebPages/affectintweets.htm) |
| nli | Adversarial NLI, round 3 test (Nie et al., 2020) | 250 | CC BY-NC 4.0 | [link](https://huggingface.co/datasets/facebook/anli) |
| nli | ChaosNLI, SNLI and MNLI parts, 100 annotations per item (Nie et al., 2020) | 250 | CC BY-NC 4.0 | [link](https://github.com/easonnie/ChaosNLI) |
| sentiment | SemEval-2018 Task 1 Affect in Tweets, valence ordinal classification (Mohammad et al., 2018) | 167 | Research use only; do not redistribute (NRC) (rebuilt by a loader) | [link](https://saifmohammad.com/WebPages/affectintweets.htm) |
| sentiment | Stanford Sentiment Treebank, 5 classes (Socher et al., 2013) | 167 | Unspecified (research use) | [link](https://huggingface.co/datasets/SetFit/sst5) |
| sentiment | TweetEval sentiment (SemEval-2017 Task 4A; Barbieri et al., 2020) | 166 | CC BY 3.0 (card) + Twitter ToS | [link](https://huggingface.co/datasets/cardiffnlp/tweet_eval) |
| spam_phishing | Enron-Spam (Metsis et al., 2006) | 167 | Not stated upstream (rebuilt by a loader) | [link](https://huggingface.co/datasets/SetFit/enron_spam) |
| spam_phishing | SMS Spam Collection (Almeida et al., 2011) | 167 | CC BY 4.0 (UCI) | [link](https://huggingface.co/datasets/ucirvine/sms_spam) |
| spam_phishing | SpamAssassin public corpus, ham and spam (2002-2005), and Jose Nazario's phishing corpus, 2004-2006 mailboxes; labels by corpus membership | 166 | SpamAssassin corpus: not stated (readme: copyright for the text remains with the original senders); Nazario phishing corpus: CC BY 4.0 | [link](https://monkey.org/~jose/phishing/) |
| stance_sarcasm_humor | Humicroedit, SemEval-2020 Task 7 subtask 1 (Hossain et al., 2020) | 167 | Not stated (research data) (rebuilt by a loader) | [link](https://www.cs.rochester.edu/u/nhossain/humicroedit.html) |
| stance_sarcasm_humor | SemEval-2016 Task 6 stance detection (Mohammad et al., 2016), via TweetEval | 166 | Research use only; do not redistribute (NRC) (rebuilt by a loader) | [link](http://www.saifmohammad.com/WebPages/StanceDataset.htm) |
| stance_sarcasm_humor | iSarcasmEval subtask A, English (Abu Farha et al., 2022) | 167 | MIT | [link](https://github.com/iabufarha/iSarcasmEval) |
| topic | 20 Newsgroups (Lang, 1995) | 167 | CC BY 4.0 (UCI) | [link](https://huggingface.co/datasets/SetFit/20_newsgroups) |
| topic | DBpedia hierarchical classes, level 2 (DeveloperOats/DBPedia_Classes) | 167 | CC0 (card); text from Wikipedia (CC BY-SA) | [link](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes) |
| topic | Web of Science WOS-46985 (Kowsari et al., 2017) | 166 | CC BY 4.0 | [link](https://huggingface.co/datasets/river-martin/web-of-science-with-label-texts) |
| toxicity | Civil Comments / Jigsaw Unintended Bias (Borkan et al., 2019) | 250 | CC0 | [link](https://huggingface.co/datasets/google/civil_comments) |
| toxicity | Dynamically Generated Hate Speech Dataset v0.2.3 (Vidgen et al., 2021) | 250 | CC BY 4.0 | [link](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) |

#### Knowledge and reasoning (6,000 requests, 30 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| answerability | MuSiQue-Full, answerable and unanswerable multi-hop questions, validation (Trivedi et al., 2022) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/bdsaglam/musique) |
| answerability | SQuAD 2.0, validation (Rajpurkar et al., 2018) | 250 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/rajpurkar/squad_v2) |
| causal | COPA, original test set with labels (Roemmele et al., 2011), via Balanced COPA | 250 | CC BY 4.0 (card); original COPA BSD-2-Clause | [link](https://huggingface.co/datasets/pkavumba/balanced-copa) |
| causal | e-CARE, validation (Du et al., 2022) | 250 | MIT | [link](https://huggingface.co/datasets/12ml/e-CARE) |
| citation_support | AttributionBench, in-distribution test (Li et al., 2024) | 250 | Apache-2.0 (component datasets keep their own licenses) | [link](https://huggingface.co/datasets/osunlp/AttributionBench) |
| citation_support | WiCE, Wikipedia citation entailment, sub-claims (Kamoi et al., 2023) | 250 | ODC-BY (annotations); Wikipedia CC BY-SA and Common Crawl terms (text) | [link](https://huggingface.co/datasets/jon-tow/wice) |
| commonsense | CommonsenseQA, validation (Talmor et al., 2019) | 167 | MIT | [link](https://huggingface.co/datasets/tau/commonsense_qa) |
| commonsense | PIQA, validation (Bisk et al., 2020) | 167 | AFL-3.0 | [link](https://huggingface.co/datasets/baber/piqa) |
| commonsense | WinoGrande XL, validation (Sakaguchi et al., 2020) | 166 | CC BY | [link](https://huggingface.co/datasets/allenai/winogrande) |
| date_reasoning | BIG-Bench Hard, date understanding (Suzgun et al., 2023) | 167 | MIT | [link](https://huggingface.co/datasets/lukaemon/bbh) |
| date_reasoning | Date arithmetic (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| date_reasoning | Timestamp comparison across time zones and formats (generated) | 166 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| fact_verification | AVeriTeC real-world claims with question-answer evidence, dev (Schlichtkrull et al., 2023) | 167 | CC BY-NC 4.0 | [link](https://huggingface.co/datasets/pminervini/averitec) |
| fact_verification | SciFact claims against paper abstracts, validation (Wadden et al., 2020) | 167 | CC BY-NC 2.0 | [link](https://huggingface.co/datasets/allenai/scifact_entailment) |
| fact_verification | VitaminC contrastive fact verification (Schuster et al., 2021) | 166 | CC BY-SA 3.0 | [link](https://huggingface.co/datasets/tals/vitaminc) |
| factual_judgment | Natural Questions open, validation (Lee et al., 2019) | 250 | CC BY-SA 3.0 | [link](https://huggingface.co/datasets/google-research-datasets/nq_open) |
| factual_judgment | SimpleQA Verified (Google, 2025; after Wei et al., 2024) | 250 | MIT | [link](https://huggingface.co/datasets/google/simpleqa-verified) |
| logic_puzzles | FOLIO v2, first-order logic reasoning in natural language, validation (Han et al., 2024) | 167 | MIT | [link](https://huggingface.co/datasets/yale-nlp/FOLIO) |
| logic_puzzles | Ordering puzzles from pairwise clues (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| logic_puzzles | ZebraLogic, multiple-choice mode (Lin et al., 2025) | 166 | Not stated | [link](https://huggingface.co/datasets/WildEval/ZebraLogic) |
| math_verification | ProcessBench model solutions (Zheng et al., 2024) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/Qwen/ProcessBench) |
| math_verification | VerifyBench, math items (Yan et al., 2025) | 250 | MIT | [link](https://huggingface.co/datasets/ZJU-REAL/VerifyBench) |
| multihop | 2WikiMultihopQA, comparison and bridge-comparison questions (Ho et al., 2020) | 167 | Apache-2.0 | [link](https://huggingface.co/datasets/framolfese/2WikiMultihopQA) |
| multihop | HotpotQA, distractor setting, comparison questions (Yang et al., 2018) | 167 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |
| multihop | StrategyQA, re-split test (Geva et al., 2021) | 166 | MIT | [link](https://huggingface.co/datasets/ChilleD/StrategyQA) |
| numeric_comparison | EQUATE quantitative NLI: AWPNLI, NewsNLI, RedditNLI (Ravichander et al., 2019) | 167 | MIT | [link](https://huggingface.co/datasets/tasksource/equate) |
| numeric_comparison | Number comparison across notations (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| numeric_comparison | Quantity comparison across units (generated) | 166 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| subject_mcq | MMLU-Pro (Wang et al., 2024) | 250 | MIT | [link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| subject_mcq | SuperGPQA (M-A-P, 2025) | 250 | ODC-BY | [link](https://huggingface.co/datasets/m-a-p/SuperGPQA) |

#### Subjective and pluralistic (1,999 requests, 14 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| moral_judgment | ETHICS commonsense morality, hard test set, short scenarios (Hendrycks et al., 2021) | 167 | MIT | [link](https://huggingface.co/datasets/hendrycks/ethics) |
| moral_judgment | Scruples Anecdotes, test split: r/AmItheAsshole posts with the community's verdict votes (Lourie et al., 2021) | 167 | Apache-2.0 | [link](https://github.com/allenai/scruples) |
| moral_judgment | Social-Chem-101, rules of thumb annotated by 50 crowd workers each (analysis split; Forbes et al., 2020) | 166 | CC BY-SA 4.0 | [link](https://github.com/mbforbes/social-chemistry-101) |
| opinion_distribution | Collective Constitutional AI: Polis votes of about 1,000 U.S. adults on rules for an AI chatbot, July 2023 (Anthropic and The Collective Intelligence Project, 2023) | 125 | Not stated | [link](https://github.com/saffronh/ccai) |
| opinion_distribution | GlobalOpinionQA: Pew Global Attitudes Survey and World Values Survey questions with answer shares by country (Durmus et al., 2023) | 125 | CC BY-NC-SA 4.0 | [link](https://huggingface.co/datasets/Anthropic/llm_global_opinions) |
| opinion_distribution | Polis open data: statements from 13 English-language public deliberations with participants' agree / disagree / pass votes (The Computational Democracy Project) | 125 | CC BY 4.0 | [link](https://github.com/compdemocracy/openData) |
| opinion_distribution | SubPOP-Eval: General Social Survey 2022 questions with answer shares of 22 U.S. subpopulations (Suh et al., 2025) | 125 | CC BY-NC-SA 4.0 (gated; research or personal non-commercial use) (rebuilt by a loader) | [link](https://huggingface.co/datasets/jjssuh/subpop) |
| perspective_labeling | DICES-350: adversarial chatbot conversations rated for safety by all 104 raters of a pool balanced by gender and race (Aroyo et al., 2023) | 167 | CC BY 4.0 | [link](https://github.com/google-research-datasets/dices-dataset) |
| perspective_labeling | DICES-990: adversarial chatbot conversations, each rated for safety by 60-70 raters balanced by gender and locale (US, India) (Aroyo et al., 2023) | 167 | CC BY 4.0 | [link](https://github.com/google-research-datasets/dices-dataset) |
| perspective_labeling | Measuring Hate Speech corpus, the 70 reference-set comments rated by 240-815 U.S. crowd workers each (Kennedy et al., 2020; Sachdeva et al., 2022) | 166 | CC BY 4.0 | [link](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |
| persuasion_humor | ExPUN: puns and non-puns from SemEval-2017 Task 7, each rated by five annotators (Sun et al., 2022) | 125 | CC BY-NC 4.0 (annotations); pun texts CC BY 4.0 or CC BY-NC 4.0 (SemEval-2017 Task 7) | [link](https://github.com/amazon-science/expunations) |
| persuasion_humor | New Yorker Cartoon Caption Contest crowd ratings (Jain et al., 2020; Zhang et al., 2024), cartoons described by the human-written annotations of Hessel et al. (2023), ranking test split | 125 | CC BY-NC 4.0 (ratings); CC BY 4.0 (cartoon descriptions) | [link](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking) |
| persuasion_humor | Persuasion dataset: claims with human- and Claude-written arguments, each read by three participants (Durmus et al., 2024) | 125 | CC BY-NC-SA 4.0 | [link](https://huggingface.co/datasets/Anthropic/persuasion) |
| persuasion_humor | UKPConvArg1, argument pairs from createdebate.com and convinceme.net with five crowd judgments each (Habernal and Gurevych, 2016) | 124 | CC BY 4.0 | [link](https://github.com/UKPLab/acl2016-convincing-arguments) |

### Applied decisions

#### Professional domains (4,500 requests, 26 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| answer_grading | ELLIPSE corpus of English-learner essays, test set (Crossley et al., 2023) | 167 | CC BY-NC-SA 4.0 (rebuilt by a loader) | [link](https://github.com/scrosseye/ELLIPSE-Corpus) |
| answer_grading | SemEval-2013 Task 7 Student Response Analysis, SciEntsBank and Beetle, 5-way (Dzikovska et al., 2013) | 166 | CC BY-SA 3.0 | [link](https://github.com/myrosia/semeval-2013-task7) |
| answer_grading | UNT Computer Science Short Answer Dataset v2.0 (Mohler, Bunescu and Mihalcea, 2011) | 167 | Not stated upstream (UNT release); CC BY 4.0 on the Hugging Face copy | [link](https://huggingface.co/datasets/nkazi/MohlerASAG) |
| document_fields | CORD v2 Consolidated Receipt Dataset, test and validation sets (Park et al., 2019) | 167 | CC BY 4.0 | [link](https://huggingface.co/datasets/naver-clova-ix/cord-v2) |
| document_fields | SROIE, ICDAR 2019 Scanned Receipts OCR and Information Extraction, test set with the 14 restored annotations (Huang et al., 2019) | 167 | CC BY 4.0 | [link](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) |
| document_fields | VRDU ad-buy forms (DeepForm political TV-ad invoices and contracts from FCC public files), mixed-template test split (Wang et al., 2023) | 166 | Not stated (documents are US public records from the FCC) | [link](https://github.com/google-research-datasets/vrdu) |
| ecommerce | Shopping Queries Dataset (ESCI), English test split (Reddy et al., 2022) | 167 | Apache-2.0 | [link](https://github.com/amazon-science/esci-data) |
| ecommerce | WANDS, Wayfair product search relevance judgements (Chen et al., 2022) | 167 | MIT | [link](https://github.com/wayfair/WANDS) |
| ecommerce | WDC-25 Gold Standard for Product Categorization, test clusters (Web Data Commons, 2019) | 166 | Not stated (Web Data Commons; offers extracted from the Common Crawl) | [link](http://webdatacommons.org/categorization/) |
| finance | Financial PhraseBank v1.0, sentences with majority agreement (Malo et al., 2014) | 167 | CC BY-NC-SA 3.0 | [link](https://huggingface.co/datasets/takala/financial_phrasebank) |
| finance | South German Credit (UPDATE), 1,000 consumer credits 1973-1975 (Groemping, 2019; corrected Statlog German Credit) | 166 | CC BY 4.0 | [link](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) |
| finance | Trillion Dollar Words: FOMC minutes, speeches and press conferences, hawkish-dovish test split (Shah et al., 2023) | 167 | CC BY-NC 4.0 | [link](https://huggingface.co/datasets/gtfintechlab/fomc_communication) |
| legal | CUAD v1, clause-type identification on the 102 test contracts (Hendrycks et al., 2021) | 167 | CC BY 4.0 | [link](https://github.com/TheAtticusProject/cuad) |
| legal | ContractNLI, document-level NLI on non-disclosure agreements, test split (Koreeda and Manning, 2021) | 167 | CC BY 4.0 | [link](https://stanfordnlp.github.io/contract-nli/) |
| legal | LegalBench, 17 classification tasks, test splits (Guha et al., 2023) | 166 | Per task: CC BY 4.0 (most), CC BY-SA 4.0 (definition_classification), CC BY-NC 4.0 (canada_tax_court_outcomes), CC BY-NC-SA 4.0 (learned_hands_*), MIT (sara_entailment) | [link](https://huggingface.co/datasets/nguha/legalbench) |
| medical | Emergency department visits with expert-reviewed KTAS triage levels (Moon et al., 2019, PLOS ONE, S1 data) | 167 | CC BY 4.0 | [link](https://doi.org/10.1371/journal.pone.0216972) |
| medical | MedQA, US (USMLE) questions with 4 options, test split (Jin et al., 2021) | 167 | MIT | [link](https://github.com/jind11/MedQA) |
| medical | PubMedQA, expert-labelled PQA-L, official 500-question test set, reasoning-required setting (Jin et al., 2019) | 166 | MIT | [link](https://github.com/pubmedqa/pubmedqa) |
| science | ICLR 2025 submissions with reviewer ratings and decisions, from OpenReview via Paper Copilot paperlists | 250 | CC BY 4.0 (ICLR 2025 submissions and reviews on OpenReview); the Paper Copilot compilation states no license | [link](https://github.com/papercopilot/paperlists) |
| science | arXiv metadata, papers first submitted in 2024 with a computer-science primary category (Cornell arXiv snapshot) | 250 | CC0 1.0 (arXiv metadata, including abstracts) | [link](https://huggingface.co/datasets/bluuebunny/arxiv_metadata_by_year) |
| secops | NSL-KDD, KDDTest+ (Tavallaee et al., 2009; from the 1998 DARPA / KDD Cup 1999 data) | 167 | Redistribution permitted with citation (UNB Canadian Institute for Cybersecurity) | [link](https://www.unb.ca/cic/datasets/nsl.html) |
| secops | NVD CVSS v3.1 base metrics assigned by NVD analysts, CVE-2025 records (NIST National Vulnerability Database) | 167 | CVE Terms of Use (descriptions, © MITRE); NVD metrics and CWE mappings are US government data | [link](https://nvd.nist.gov/vuln/data-feeds) |
| secops | NVD CWE mappings assigned by NVD analysts, CVE-2025 records (NIST National Vulnerability Database) | 166 | CVE Terms of Use (descriptions, © MITRE); NVD metrics and CWE mappings are US government data; CWE names and descriptions © MITRE (CWE Terms of Use) | [link](https://nvd.nist.gov/vuln/categories) |
| support_tickets | BANKING77, online banking customer queries, test split (Casanueva et al., 2020) | 167 | CC BY 4.0 | [link](https://github.com/PolyAI-LDN/task-specific-datasets) |
| support_tickets | CFPB Consumer Complaint Database, narratives of complaints received in 2022 (via a Hugging Face snapshot) | 167 | CC0 1.0 (US government public data) | [link](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints) |
| support_tickets | Mozilla Bugzilla defect reports filed January-June 2024 and later fixed (bugzilla.mozilla.org) | 166 | Not stated (public bug reports; Mozilla Websites & Communications Terms of Use) | [link](https://bugzilla.mozilla.org) |

#### Retrieval, ranking and matching (2,499 requests, 13 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| context_sufficiency | DuoRC ParaphraseRC, test (Saha et al., 2018) | 167 | MIT | [link](https://huggingface.co/datasets/ibm-research/duorc) |
| context_sufficiency | IIRC, incomplete information reading comprehension, test (Ferguson et al., 2020) | 166 | CC BY 4.0 | [link](https://allenai.org/data/iirc) |
| context_sufficiency | MS MARCO question answering v2.1, dev set (Bajaj et al., 2018) | 166 | MS MARCO terms: non-commercial research use | [link](https://huggingface.co/datasets/microsoft/ms_marco) |
| entity_matching | Abt-Buy (Köpcke, Thor and Rahm, 2010), DeepMatcher test split (Mudgal et al., 2018) | 167 | Not stated (Leipzig entity-resolution benchmark) | [link](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#abt-buy) |
| entity_matching | DBLP-Scholar (Köpcke, Thor and Rahm, 2010), DeepMatcher test split (Mudgal et al., 2018) | 167 | Not stated (Leipzig entity-resolution benchmark) | [link](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#dblp-scholar) |
| entity_matching | WDC Products, 80% corner cases, seen-products test set (Peeters, Der and Bizer, 2024) | 166 | Not stated (offers extracted from the Common Crawl) | [link](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) |
| relevance_grading | ANTIQUE non-factoid answer retrieval, test judgments (Hashemi et al., 2020) | 167 | Research use (ANTIQUE readme); answers from Yahoo! Answers (Webscope L6), whose terms do not allow redistribution (rebuilt by a loader) | [link](https://ciir.cs.umass.edu/downloads/Antique/) |
| relevance_grading | TREC 2019 and 2020 Deep Learning Track, passage judgments on MS MARCO (Craswell et al., 2020; 2021) | 166 | MS MARCO terms: non-commercial research use; NIST relevance judgments | [link](https://microsoft.github.io/msmarco/TREC-Deep-Learning) |
| relevance_grading | TREC-COVID, final cumulative judgments (Voorhees et al., 2020); article texts from the BEIR corpus (Thakur et al., 2021) | 167 | CORD-19 Dataset License Agreement (text and data mining only, non-transferable); judgments by NIST (rebuilt by a loader) | [link](https://ir.nist.gov/trec-covid/) |
| reranking | BRIGHT, StackExchange domains (Su et al., 2025) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/xlangai/BRIGHT) |
| reranking | TREC 2019 and 2020 Deep Learning Track, passage judgments on MS MARCO (Craswell et al., 2020; 2021) | 250 | MS MARCO terms: non-commercial research use; NIST relevance judgments | [link](https://microsoft.github.io/msmarco/TREC-Deep-Learning) |
| retrieval_need | FreshQA, test questions, snapshot of 2026-04-21 (Vu et al., 2024) | 250 | Apache-2.0 | [link](https://github.com/freshllms/freshqa) |
| retrieval_need | SituatedQA, temporal context-dependent question identification, test (Zhang and Choi, 2021) | 250 | Not stated (SituatedQA annotations); questions from WebQuestions (CC BY 4.0) and MS MARCO (non-commercial research terms) | [link](https://github.com/mikejqzhang/SituatedQA) |

#### Structured data, tables and logs (2,500 requests, 13 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| column_types | SOTAB V2, WDC Schema.org Table Annotation Benchmark, column type annotation test set (Korini, Peeters and Bizer, 2023) | 167 | Not stated (Web Data Commons; data extracted from the Common Crawl) | [link](http://webdatacommons.org/structureddata/sotab/v2/) |
| column_types | WDC Schema Matching Benchmark, SOTAB-SM (100-table variant), test split (Der and Bizer, 2023) | 167 | Not stated (Web Data Commons; data extracted from the Common Crawl) | [link](http://webdatacommons.org/structureddata/smb/) |
| column_types | WDC Schema Matching Benchmark, T2D-SM with column headers, test split (Der and Bizer, 2023; built on the T2Dv2 gold standard, Ritze et al., 2015) | 166 | Not stated (Web Data Commons; data extracted from the Common Crawl); DBpedia-derived tables CC BY-SA 3.0 | [link](http://webdatacommons.org/structureddata/smb/) |
| data_quality_rules | Data-quality rules over generated tables (generated) | 250 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| data_quality_rules | HoloClean hospital data with injected typos and its denial constraints (Rekatsinas et al., 2017) | 250 | Apache-2.0 (HoloClean repository); hospital data from US CMS Hospital Compare | [link](https://github.com/HoloClean/holoclean/tree/master/testdata) |
| log_anomaly | Loghub BGL, Blue Gene/L RAS log with alert tags from the administrators' rules (Oliner and Stearley, 2007) | 167 | CC BY 4.0 (Loghub, Zenodo record 8196385) | [link](https://github.com/logpai/loghub/tree/master/BGL) |
| log_anomaly | Loghub BGL, alert categories assigned by the administrators' rules (Oliner and Stearley, 2007) | 167 | CC BY 4.0 (Loghub, Zenodo record 8196385) | [link](https://github.com/logpai/loghub/tree/master/BGL) |
| log_anomaly | Loghub HDFS_v1, block-level traces labelled normal or anomalous (Xu et al., 2009) | 166 | CC BY 4.0 (Loghub, Zenodo record 8196385) | [link](https://github.com/logpai/loghub/tree/master/HDFS) |
| sql_verification | NL2SQL-BUGs, BIRD dev questions with SQL from RESDSQL, CodeS and CHESS (Liu et al., 2025) | 250 | Not stated (NL2SQL-BUGs labels); BIRD questions, evidence and schemas CC BY-SA 4.0 | [link](https://github.com/HKUSTDial/NL2SQL-Bugs-Benchmark) |
| sql_verification | Spider dev with beam predictions of SmBoP, RESDSQL+NatSQL and NatSQL (Yu et al., 2018; Chen et al., 2023) | 250 | Spider CC BY-SA 4.0; parser predictions: not stated | [link](https://github.com/OSU-NLP-Group/Text2SQL-Error-Detection) |
| table_qa | TAT-QA, test set with released gold answers (Zhu et al., 2021) | 167 | CC BY 4.0 | [link](https://github.com/NExTplusplus/TAT-QA) |
| table_qa | TabMWP, multiple-choice questions of the test split (Lu et al., 2023) | 167 | CC BY-NC-SA 4.0 | [link](https://github.com/lupantech/PromptPG) |
| table_qa | WikiTableQuestions, pristine-unseen-tables test set (Pasupat and Liang, 2015) | 166 | CC BY-SA 4.0 | [link](https://github.com/ppasupat/WikiTableQuestions) |

#### Code (3,886 requests, 17 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| code_search | CoSQA, Bing web queries paired with Python functions, dev split (Huang et al., 2021) | 250 | MIT (CoCLR repository); C-UDA 1.0 as part of CodeXGLUE; functions from the CodeSearchNet corpus | [link](https://github.com/Jun-jie-Huang/CoCLR) |
| code_search | CodeSearchNet Challenge expert relevance annotations, six languages (Husain et al., 2019) | 250 | MIT (annotations); code from public GitHub repositories under their own licenses | [link](https://github.com/github/CodeSearchNet) |
| commit_classification | CCS conventional commit classification dataset, test split (Zeng et al., ICSE 2025) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/0x404/ccs_dataset) |
| commit_classification | CodeReviewer diff quality estimation, test split (Li et al., FSE 2022) | 250 | CC BY 4.0 (Zenodo record); code from open-source GitHub projects | [link](https://zenodo.org/records/6900648) |
| fault_localization | SWE-bench test set, file-level localization from the gold patch (Jimenez et al., 2024) | 167 | MIT (SWE-bench repository) | [link](https://huggingface.co/datasets/princeton-nlp/SWE-bench) |
| fault_localization | SWE-bench test set, function-level localization inside the fixed file, oracle file contents (Jimenez et al., 2024) | 167 | MIT (SWE-bench repository) | [link](https://huggingface.co/datasets/princeton-nlp/SWE-bench_oracle) |
| fault_localization | SWE-bench-Live, issues from 2024-2025, file-level localization from the gold patch (Zhang et al., 2025) | 166 | MIT | [link](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live) |
| output_prediction | CRUXEval-O, output prediction for 800 short Python functions (Gu et al., 2024) | 250 | MIT | [link](https://github.com/facebookresearch/cruxeval) |
| output_prediction | LiveCodeBench code execution v2, LeetCode solutions with recorded outputs (Jain et al., 2024) | 250 | Not stated precisely (the Hugging Face card says 'cc'); LiveCodeBench code and submissions MIT | [link](https://huggingface.co/datasets/livecodebench/execution-v2) |
| patch_correctness | SWE-bench-extra tasks with SWE-agent final patches from Nebius's open-weight agent runs and their test results (Golubev et al., 2024) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) |
| patch_correctness | SWE-rebench tasks with OpenHands + Qwen3-Coder-480B final patches and their test results (Badertdinov et al., 2025) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) |
| static_analysis_triage | D2A leaderboard, Trace + Code task, dev split: Infer warnings on C projects labelled by differential analysis (Zheng et al., ICSE-SEIP 2021) | 250 | Apache-2.0 (as listed by the Hugging Face copy claudios/D2A; the IBM D2A repository is Apache-2.0) | [link](https://github.com/IBM/D2A) |
| static_analysis_triage | OWASP Benchmark for Java v1.2: 2,740 servlets, each a true or a false positive for one vulnerability category (OWASP Foundation) | 250 | GPL-2.0 | [link](https://github.com/OWASP-Benchmark/BenchmarkJava) |
| test_outcome | HumanEvalFix from HumanEvalPack, correct and buggy solutions with their unit tests, six languages (Muennighoff et al., 2023) | 250 | MIT | [link](https://huggingface.co/datasets/bigcode/humanevalpack) |
| test_outcome | LiveCodeBench code generation, recorded per-test results of Gemini-Pro-1.5 programs (Jain et al., 2024) | 136 | MIT | [link](https://huggingface.co/datasets/livecodebench/submissions) |
| vulnerability | PrimeVul v0.1, paired test set: vulnerable and patched C/C++ functions (Ding et al., ICSE 2025) | 250 | MIT | [link](https://github.com/DLVulDet/PrimeVul) |
| vulnerability | SecVulEval, C/C++ CVE fixes from NVD, vulnerable and fixed function pairs (Ahmed et al., 2025) | 250 | MIT | [link](https://huggingface.co/datasets/arag0rn/SecVulEval) |

#### Forecasting (1,420 requests, 7 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| binary_events | Binary forecasting questions from Metaculus, Manifold, Polymarket, Good Judgment Open and INFER, test split (Halawi et al., 2024) | 166 | Apache-2.0 | [link](https://huggingface.co/datasets/YuehHanChen/forecasting) |
| binary_events | ForecastBench question sets, prediction-market and forecasting-platform questions (Karger et al., 2025) | 167 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets) |
| binary_events | ForecastBench question sets, questions resolved from FRED, DBnomics, Wikipedia and Yahoo Finance data (Karger et al., 2025) | 167 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets) |
| multi_outcome | Polymarket multi-outcome events ended January-August 2026 (gamma API) | 250 | Polymarket Terms of Use (content reserved by Polymarket); not redistributed (rebuilt by a loader) | [link](https://docs.polymarket.com/developers/gamma-markets-api/overview) |
| multi_outcome | Which related daily FRED series moves most (currencies against the dollar, energy prices, Treasury yields), resolved by September 2026 | 170 | Public-domain series from the Federal Reserve Board and the U.S. EIA via FRED; items generated by S1-Arena (CC BY 4.0) | [link](https://fred.stlouisfed.org/) |
| numeric_bins | Daily FRED series (Treasury yields, breakeven inflation, exchange rates, oil and gas prices), values published up to September 2026 | 250 | Public-domain series from the Federal Reserve Board and the U.S. EIA via FRED; items generated by S1-Arena (CC BY 4.0) | [link](https://fred.stlouisfed.org/) |
| numeric_bins | Polymarket range ('bracket') events ended January-August 2026 (gamma API) | 250 | Polymarket Terms of Use (content reserved by Polymarket); not redistributed (rebuilt by a loader) | [link](https://docs.polymarket.com/developers/gamma-markets-api/overview) |

#### Dialogue and interaction (2,464 requests, 10 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| clarification_need | AmbigQA / AmbigNQ, validation (Min et al., 2020) | 250 | CC BY-SA 3.0 | [link](https://huggingface.co/datasets/sewon/ambig_qa) |
| clarification_need | ShARC conversational machine reading with rules, test (Saeidi et al., 2018) | 250 | CC BY-SA 3.0 | [link](https://sharc-data.github.io/) |
| end_of_turn | Switchboard Dialog Act Corpus (SwDA), test conversations of Stolcke et al. (2000) (Jurafsky et al., 1997) | 250 | CC BY-NC-SA 3.0 | [link](https://github.com/cgpotts/swda) |
| end_of_turn | Taskmaster-1 spoken two-person (Wizard-of-Oz) dialogues: user turns transcribed from speech (Byrne et al., 2019) | 250 | CC BY 4.0 | [link](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019) |
| human_handoff | ABCD, Action-Based Conversations Dataset: customer-service chats escalated to a manager, all splits (Chen et al., 2021) | 220 | MIT | [link](https://github.com/asappresearch/abcd) |
| human_handoff | Scope cards over human intent labels: HINT3 v2 test sets, real queries to three chatbots (Arora et al., 2020), and NatCS customer-service requests in insurance, banking and finance (Gung et al., 2023) | 250 | HINT3: ODbL 1.0 (database) and DbCL 1.0 (contents); NatCS (DSTC11 track 2 data): CDLA-Permissive-2.0 | [link](https://github.com/hellohaptik/HINT3) |
| state_tracking | MultiWOZ 2.2, test split (Zang et al., 2020; Budzianowski et al., 2018) | 249 | MIT | [link](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2) |
| state_tracking | Schema-Guided Dialogue (SGD / DSTC8), test split (Rastogi et al., 2020) | 245 | CC BY-SA 4.0 | [link](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) |
| user_satisfaction | EmoWOZ, test split: MultiWOZ and DialMAGE human-machine dialogues (Feng et al., 2022) | 250 | CC BY-NC 4.0 | [link](https://zenodo.org/record/6506504) |
| user_satisfaction | User Satisfaction Simulation (USS), exchange-level ratings on MultiWOZ, SGD, ReDial and CCPE dialogues (Sun et al., 2021) | 250 | Not stated (source dialogues: MultiWOZ MIT, SGD CC BY-SA 4.0, ReDial CC BY 4.0, CCPE CC BY-SA 4.0) | [link](https://github.com/sunnweiwei/user-satisfaction-simulation) |

#### Batch semantic processing (1,500 requests, 6 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| semantic_filter | EMSCAD, Employment Scam Aegean Dataset: 17,880 job ads from Workable, 866 of them fraudulent (Vidros et al., 2017) | 250 | CC0-1.0 (Kaggle release) | [link](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| semantic_filter | Women's E-Commerce Clothing Reviews, 23,486 reviews from an online retailer (nicapotato, Kaggle) | 250 | CC0-1.0 | [link](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) |
| semantic_grouping | Amazon Berkeley Objects (ABO) product listings, English listings of amazon.com, .co.uk and .ca (Collins et al., CVPR 2022) | 250 | CC BY 4.0 | [link](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| semantic_grouping | Twitter US Airline Sentiment, negative tweets with the complaint reason (Figure Eight / CrowdFlower Data for Everyone, 2015) | 250 | CC BY-NC-SA 4.0 | [link](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment) |
| semantic_ranking | TREC 2022 Clinical Trials track: synthetic patient cases and ClinicalTrials.gov records with NIST judgments (Roberts et al., 2022) | 250 | CC BY-SA 4.0 (Hugging Face copy); trial records from ClinicalTrials.gov (public domain) | [link](https://huggingface.co/datasets/araag2/TREC_Clinical-Trials) |
| semantic_ranking | TripAdvisor hotel reviews with the reviewers' aspect ratings, English reviews of 2012 (Li et al., 2013; processed by Niimi, 2024) | 250 | Apache-2.0 (Hugging Face release); original crawl by Li, Ott and Cardie (2013), terms not stated | [link](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating) |

### Evaluation and control

#### LLM-as-judge and evaluators (4,998 requests, 24 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| agent_trajectory | AgentRewardBench, expert-annotated web agent trajectories, test split (Lù et al., 2025) | 250 | AgentRewardBench terms of use: research use under fair use; derivatives must include these terms (no license file; trajectories contain third-party web content) | [link](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) |
| agent_trajectory | tau-bench historical trajectories of GPT-4o and Claude 3.5 Sonnet (new) in the airline and retail domains (Yao et al., 2024) | 250 | MIT | [link](https://github.com/sierra-research/tau-bench) |
| atomic_factuality | FActScore human-annotated biographies by InstructGPT, ChatGPT and PerplexityAI (Min et al., 2023) | 250 | MIT | [link](https://github.com/shmsw25/FActScore) |
| atomic_factuality | FELM, segment-level factuality of ChatGPT responses (Chen et al., 2023) | 250 | CC BY-NC-SA 4.0 | [link](https://huggingface.co/datasets/hkust-nlp/felm) |
| instruction_following | IFBench test prompts (Pyatkin et al., 2025) with model responses and strict checker results from shisa-ai/eval-IFBench-results | 250 | ODC-BY (results); IFBench prompts ODC-BY | [link](https://huggingface.co/datasets/shisa-ai/eval-IFBench-results) |
| instruction_following | IFEval prompts (Zhou et al., 2023) with sampled responses and strict checker results from PPE (Frick et al., 2025) | 250 | Apache-2.0 (prompts); model outputs under the providers' terms of use (PPE dataset card) | [link](https://huggingface.co/datasets/lmarena-ai/PPE-IFEval-Best-of-K) |
| pairwise_preference | JudgeBench, GPT-4o and Claude-3.5-Sonnet response pairs (Tan et al., 2025) | 166 | MIT | [link](https://huggingface.co/datasets/ScalerLab/JudgeBench) |
| pairwise_preference | RM-Bench, chat, math and code domains (Liu et al., 2025) | 166 | ODC-BY | [link](https://huggingface.co/datasets/THU-KEG/RM-Bench) |
| pairwise_preference | RewardBench 2, Math, Precise IF, Focus and Ties subsets (Malik et al., 2025) | 166 | ODC-BY | [link](https://huggingface.co/datasets/allenai/reward-bench-2) |
| pointwise_rating | BiGGen Bench, human evaluation split (Kim et al., 2025) | 250 | CC BY-SA 4.0 (BiGGen Bench); results repository: not stated | [link](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results) |
| pointwise_rating | HelpSteer2, validation split with individual annotator ratings (Wang et al., 2024) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| process_reward | MR-Ben meta-reasoning benchmark, all subjects except coding (Zeng et al., 2024) | 167 | MIT | [link](https://huggingface.co/datasets/Randolphzeng/Mr-Ben) |
| process_reward | PRM800K, phase 2 test split (Lightman et al., 2024) | 167 | MIT | [link](https://github.com/openai/prm800k) |
| process_reward | ProcessBench, earliest-error paragraph (Zheng et al., 2025) | 166 | Apache-2.0 | [link](https://huggingface.co/datasets/Qwen/ProcessBench) |
| rag_faithfulness | BEGIN, knowledge-grounded dialogue responses, test split (Dziri et al., 2022) | 250 | CC BY 4.0 (WoW and CMU-DoG parts); CDLA-Sharing 1.0 (Topical-Chat part) | [link](https://github.com/google/BEGIN-dataset) |
| rag_faithfulness | RAGTruth, test split: QA (MS MARCO), data-to-text (Yelp) and news summarization (CNN/DM, Recent News) (Niu et al., 2024) | 250 | MIT | [link](https://github.com/ParticleMedia/RAGTruth) |
| reference_answer | Answer Equivalence (AE) dataset, test split, SQuAD questions (Bulian et al., 2022) | 167 | Apache-2.0 (questions and passages from SQuAD, CC BY-SA 4.0) | [link](https://github.com/google-research-datasets/answer-equivalence-dataset) |
| reference_answer | EVOUNA, Natural Questions and TriviaQA answers of five QA systems judged by humans (Wang et al., 2023) | 167 | Apache-2.0 | [link](https://github.com/wangcunxiang/QA-Eval) |
| reference_answer | VerifierBench (CompassVerifier; Liu et al., 2025) | 166 | CC BY-SA 4.0 | [link](https://huggingface.co/datasets/opencompass/VerifierBench) |
| summary_consistency | FaithBench, summaries by 10 modern LLMs (Bao et al., 2025) | 167 | CC BY-NC-SA 4.0 | [link](https://github.com/vectara/FaithBench) |
| summary_consistency | SummEdits, 10 domains (Laban et al., 2023) | 167 | CC BY 4.0 | [link](https://huggingface.co/datasets/Salesforce/summedits) |
| summary_consistency | TofuEval, MeetingBank part, test split (Tang et al., 2024) | 166 | MIT-0 (annotations); MeetingBank transcripts CC BY-NC-SA 4.0 | [link](https://github.com/amazon-science/tofueval) |
| translation_quality | WMT 2020 newstest MQM annotations, en-de and zh-en (Freitag et al., 2021) | 250 | Apache-2.0 (annotations); WMT news test data for research use | [link](https://github.com/google/wmt-mqm-human-evaluation) |
| translation_quality | WMT 2023 general MT MQM annotations with three ratings per segment, en-de and zh-en (Kocmi et al., 2023; Google MQM release) | 250 | Apache-2.0 (annotations); WMT general test data for research use | [link](https://github.com/google/wmt-mqm-human-evaluation) |

#### Safety and moderation (2,998 requests, 16 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| agent_action_risk | ASSEBench safety and security subsets, unambiguous records (Luo et al., 2025, AgentAuditor) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/Atarogic/ASSEBench) |
| agent_action_risk | ATBench agent trajectory safety benchmark, 1,000-trajectory release (Li et al., 2026) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/AI45Research/ATBench) |
| custom_policy | CoSApien, human-authored safety configs and test prompts (Zhang et al., ICLR 2025, Controllable Safety Alignment) | 167 | CDLA-Permissive-2.0 | [link](https://huggingface.co/datasets/microsoft/CoSApien) |
| custom_policy | DynaBench test set of the DynaGuard project: business policies and user-agent dialogues (Hoover et al., 2025) | 167 | MIT | [link](https://huggingface.co/datasets/tomg-group-umd/DynaBench) |
| custom_policy | Policy variants over the OpenAI moderation evaluation set (texts and human labels from Markov et al., 2023; policies generated) | 166 | Not stated (the release repository has no license file) | [link](https://github.com/openai/moderation-api-release) |
| harmfulness | Aegis 2.0 / Nemotron Content Safety Dataset V2, test, user prompts (Ghosh et al., 2025) | 167 | CC BY 4.0 | [link](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) |
| harmfulness | BeaverTails-330k, test, question-answer pairs with three annotations each (Ji et al., 2023) | 167 | CC BY-NC 4.0 | [link](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) |
| harmfulness | WildGuardTest, prompt and response harmfulness (Han et al., 2024) | 166 | ODC-BY (AI2 Responsible Use Guidelines; gated) | [link](https://huggingface.co/datasets/allenai/wildguardmix) |
| injection | AgentDojo tool-use trajectories of GPT-4o, clean and with injections in tool outputs (Debenedetti et al., 2024) | 166 | MIT | [link](https://github.com/ethz-spylab/agentdojo) |
| injection | BIPIA indirect prompt injection benchmark: email, table and code contexts with test attacks (Yi et al., 2025) | 166 | MIT (BIPIA); contexts: OpenAI Evals emails (MIT), WikiTableQuestions and Stack Exchange (CC BY-SA 4.0) | [link](https://github.com/microsoft/BIPIA) |
| injection | ToxicChat 0124, test, human-annotated user prompts with the jailbreaking label (Lin et al., 2023) | 166 | CC BY-NC 4.0 | [link](https://huggingface.co/datasets/lmsys/toxic-chat) |
| pii | Customer messages, forms and log lines with inserted personal information (generated) | 250 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| pii | Text Anonymization Benchmark (TAB), test split: ECHR court judgments annotated for anonymization (Pilán et al., 2022) | 250 | MIT | [link](https://github.com/NorskRegnesentral/text-anonymization-benchmark) |
| refusal | CoCoNot, human-verified noncompliance test set and contrast set (Brahman et al., 2024) | 167 | ODC-BY (repository LICENSE.md); the card also cites the AI2 ImpACT-LR license | [link](https://huggingface.co/datasets/allenai/coconot) |
| refusal | XSTest v2 model completions with two human annotations each (Röttger et al., 2024) | 166 | CC BY 4.0 | [link](https://github.com/paul-rottger/xstest/tree/main/model_completions) |
| refusal | XSTest v2 prompts: 250 safe prompts and 200 unsafe contrast prompts (Röttger et al., 2024) | 167 | CC BY 4.0 | [link](https://github.com/paul-rottger/xstest) |

#### Routing and orchestration (3,462 requests, 18 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| context_pruning | Leave-one-out paragraph ablation on generated multi-hop fact chains (generated) | 250 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| context_pruning | QASPER, questions on NLP papers with annotated evidence paragraphs, test split (Dasigi et al., 2021) | 250 | CC BY 4.0 | [link](https://huggingface.co/datasets/allenai/qasper) |
| difficulty | BigCodeBench-Instruct tasks with leaderboard solve rates (Zhuo et al., 2025) | 167 | Apache-2.0 | [link](https://huggingface.co/datasets/bigcode/bigcodebench-solve-rate) |
| difficulty | EmbedLLM correctness matrix, test questions (Zhuang et al., 2024) | 167 | Apache-2.0 (card); source benchmarks keep their own licenses | [link](https://huggingface.co/datasets/RZ412/EmbedLLM) |
| difficulty | LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025) | 166 | Not stated (dataset cards); code Apache-2.0. Texts from LeetCode/AtCoder via LiveCodeBench, arXiv abstracts and NYT Connections | [link](https://huggingface.co/datasets/livebench/model_judgment) |
| escalation | AutoMix cascade outputs on QuALITY, validation (Aggarwal et al., 2024) | 167 | Apache-2.0 (AutoMix); QuALITY CC BY 4.0 | [link](https://github.com/automix-llm/automix) |
| escalation | LiveBench model answers and judgments: GPT-4o mini drafts, o1-preview as the stronger model (White et al., 2025) | 167 | Not stated (dataset cards); code Apache-2.0. Texts from LeetCode/AtCoder via LiveCodeBench, arXiv abstracts and NYT Connections | [link](https://huggingface.co/datasets/livebench/model_answer) |
| escalation | RouterArena cached GPT-4o mini replies, full split (Lu et al., 2025) | 166 | Apache-2.0 (RouterArena repository); source datasets keep their own licenses | [link](https://github.com/RouteWorks/RouterArena) |
| memory_decisions | DuLeMon, Chinese long-term persona memory dialogues, 'both' test split (Xu et al., 2022) | 250 | Not stated (data); code Apache-2.0 | [link](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2022-DuLeMon) |
| memory_decisions | Multi-Session Chat, per-turn memory summaries, test split (Xu et al., 2022) | 250 | Not stated (data); ParlAI code MIT | [link](https://parl.ai/projects/msc/) |
| model_correctness | LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025) | 167 | Not stated (dataset cards); code Apache-2.0. Texts from LeetCode/AtCoder via LiveCodeBench, arXiv abstracts and NYT Connections | [link](https://huggingface.co/datasets/livebench/model_judgment) |
| model_correctness | RouterArena sub_10 split with cached replies of three models (Lu et al., 2025) | 167 | Apache-2.0 (RouterArena repository); source datasets keep their own licenses | [link](https://github.com/RouteWorks/RouterArena) |
| model_correctness | RouterBench, zero-shot MMLU, HellaSwag and ARC-Challenge prompts (Hu et al., 2024) | 166 | Not stated (dataset card); code repository MIT | [link](https://huggingface.co/datasets/withmartian/routerbench) |
| model_selection | LiveBench code generation, code completion, typo fixing and Connections puzzles (White et al., 2025) | 212 | Not stated (dataset cards); code Apache-2.0. Texts from LeetCode/AtCoder via LiveCodeBench, arXiv abstracts and NYT Connections | [link](https://huggingface.co/datasets/livebench/model_judgment) |
| model_selection | RouterBench, zero-shot MMLU, HellaSwag and ARC-Challenge prompts (Hu et al., 2024) | 250 | Not stated (dataset card); code repository MIT | [link](https://huggingface.co/datasets/withmartian/routerbench) |
| tool_selection | Berkeley Function Calling Leaderboard v4, expert-written 'multiple' and 'irrelevance' categories (Patil et al., 2025) | 167 | Apache-2.0 | [link](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| tool_selection | Berkeley Function Calling Leaderboard v4, user-contributed 'live_multiple' and 'live_irrelevance' categories (Patil et al., 2025) | 167 | Apache-2.0 | [link](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| tool_selection | MetaTool ToolE, tool selection with and without the right tool in the list (Huang et al., 2024) | 166 | MIT | [link](https://github.com/HowieHwong/MetaTool) |

## Agentic tasks

### Static decisions

The paper compares models on the agent single steps. The static game and embodied decisions are part of the benchmark but not of the paper's model comparison.

#### Agent single steps (3,481 requests, 18 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| action_critic | AgentProcessBench, HotpotQA and GAIA (text) subsets, human step labels (2026) | 167 | MIT | [link](https://huggingface.co/datasets/LulaCola/AgentProcessBench) |
| action_critic | TRBench-BFCL, single-turn parallel and parallel-multiple categories (Li et al., 2025, ToolRM) | 167 | Apache-2.0 (Hugging Face card and LICENSE file; the GitHub README says CC BY-NC-SA 4.0) | [link](https://huggingface.co/datasets/RioLee/TRBench-BFCL) |
| action_critic | WebRewardBench, Mind2Web test steps with expert and model-proposed actions (Chae et al., 2025, Web-Shepherd) | 166 | Not stated (WebRewardBench); built on the Mind2Web test set (CC BY 4.0), whose authors ask not to redistribute it (rebuilt by a loader) | [link](https://huggingface.co/datasets/WebShepherd/WebRewardBench) |
| completion | Terminal-Bench 2.0 leaderboard runs of the Terminus-2 agent with eight models (Merrill et al., 2026) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) |
| completion | tau2-bench telecom domain in solo mode (no user), published simulations of GPT-4.1 and o4-mini agents (Barres et al., 2025) | 250 | MIT | [link](https://github.com/sierra-research/tau2-bench) |
| failure_attribution | AgentErrorBench, failed ALFWorld, WebShop and GAIA runs with expert root-cause annotation (Zhu et al., 2025, AgentDebug) | 167 | MIT (repository license; the data is linked from the repository README) | [link](https://github.com/ulab-uiuc/AgentDebug) |
| failure_attribution | TRAIL, expert-annotated error taxonomy over GAIA and SWE-bench agent traces (Deshpande et al., 2025) | 148 | MIT, gated: users agree not to reshare the dataset outside a gated or private repository (rebuilt by a loader) | [link](https://huggingface.co/datasets/PatronusAI/TRAIL) |
| failure_attribution | Who&When, failed multi-agent logs on GAIA and AssistantBench with annotated failure agent and step (Zhang et al., 2025) | 166 | MIT (GitHub repository ag2ai/Agents_Failure_Attribution; the Hugging Face card states no license) | [link](https://huggingface.co/datasets/Kevin355/Who_and_When) |
| next_action | AndroidControl official test split, human demonstrations on 833 Android apps (Li et al., 2024) | 250 | Apache-2.0 (google-research repository); read through public conversions (ckg/AndroidControlParsed-20k: MIT; HarrytheOrange/parsed_AndroidControl: Apache-2.0) | [link](https://github.com/google-research/google-research/tree/master/android_control) |
| next_action | Mind2Web test splits (cross-task, cross-website, cross-domain) in the human-verified Multimodal-Mind2Web release (Deng et al., 2023) | 250 | CC BY 4.0 (Mind2Web; the authors ask not to redistribute the unzipped test data); the Multimodal-Mind2Web card lists OpenRAIL (rebuilt by a loader) | [link](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) |
| progress_success | SWE-rebench July 2026 leaderboard trajectories: 111 fresh issues, agentic systems and models in a common scaffold (Nebius, 2026) | 167 | Other: the release does not relicense third-party content; task text and repository content stay under their repositories' licenses, agent outputs under the agents' and model providers' terms | [link](https://huggingface.co/datasets/ibragim-bad/swe_rebench_07_2026_trajectories) |
| progress_success | Terminal-Bench 2.0 leaderboard runs of the Terminus-2 agent with eight models (Merrill et al., 2026) | 166 | Apache-2.0 | [link](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) |
| progress_success | tau2-bench telecom domain, published simulations of four agent models with a GPT-4.1 user simulator (Barres et al., 2025) | 167 | MIT | [link](https://github.com/sierra-research/tau2-bench) |
| step_correctness | TRBench-BFCL, multi-turn base category (Li et al., 2025, ToolRM), built on BFCL v3 multi-turn | 250 | Apache-2.0 (Hugging Face card and LICENSE file; the GitHub README says CC BY-NC-SA 4.0) | [link](https://huggingface.co/datasets/RioLee/TRBench-BFCL) |
| step_correctness | ToolComp process-supervision pairs, human-corrected versus original ReAct steps and action plans (Nath et al., 2025) | 250 | Not stated (the sibling repository vaskarnath/toolcomp is Apache-2.0) | [link](https://huggingface.co/datasets/vaskarnath/toolcomp_process_supervision_eval) |
| tool_arguments | Berkeley Function Calling Leaderboard v4, expert-written 'simple' categories in Python, Java and JavaScript (Patil et al., 2025) | 167 | Apache-2.0 | [link](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| tool_arguments | Berkeley Function Calling Leaderboard v4, user-contributed 'live_simple' category (Patil et al., 2025) | 167 | Apache-2.0 | [link](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| tool_arguments | ComplexFuncBench, multi-step Booking.com function calling with recorded API results (Zhong et al., 2025) | 166 | Apache-2.0 | [link](https://huggingface.co/datasets/zai-org/ComplexFuncBench) |

#### Games (static) (2,500 requests, 13 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| chess_best_move | ChessBench test positions with Stockfish 16 action values (Ruoss et al., NeurIPS 2024) | 250 | CC BY 4.0 (ChessBench annotations); positions from the Lichess database (CC0) | [link](https://github.com/google-deepmind/searchless_chess) |
| chess_best_move | Lichess puzzle database, first shard of the September 2026 export (lichess.org) | 250 | CC0 1.0 | [link](https://huggingface.co/datasets/Lichess/chess-puzzles) |
| poker | PokerBench post-flop test set, solutions from WASM-Postflop (Zhuang et al., AAAI 2025) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/RZ412/PokerBench) |
| poker | PokerBench pre-flop test set, solutions from GTO Wizard charts (Zhuang et al., AAAI 2025) | 250 | Apache-2.0 | [link](https://huggingface.co/datasets/RZ412/PokerBench) |
| position_eval | ChessBench test positions with Stockfish 16 state values (Ruoss et al., NeurIPS 2024) | 167 | CC BY 4.0 (ChessBench annotations); positions from the Lichess database (CC0) | [link](https://github.com/google-deepmind/searchless_chess) |
| position_eval | Connect-4 opening database: all 8-ply positions with their game-theoretic value (John Tromp, UCI repository) | 167 | CC BY 4.0 | [link](https://archive.ics.uci.edu/dataset/26/connect+4) |
| position_eval | Lichess rated standard games of 1 June 2024, first shard of the June 2024 export (lichess.org) | 166 | CC0 1.0 | [link](https://huggingface.co/datasets/Lichess/standard-chess-games) |
| puzzles | 8-puzzle positions with a unique best first move (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| puzzles | Minesweeper Beginner positions (9x9, 10 mines) with an exact deduction per square (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| puzzles | Sudoku grids: rule check and completability (generated) | 166 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| social_games | AH2AC2 human Hanabi games from hanab.live, 2- and 3-player (Dizdarević et al., 2025), with exact card-counting beliefs | 167 | Apache-2.0 | [link](https://huggingface.co/datasets/ah2ac2/datasets) |
| social_games | FanLang-9: human 9-player Werewolf games with roles, votes and transcribed speeches (Wu et al., 2024) | 166 | MIT | [link](https://github.com/boluoweifenda/werewolf) |
| social_games | It Takes Two to Lie: Diplomacy negotiation messages with the senders' own truthfulness labels, test games (Peskov et al., ACL 2020) | 167 | CC BY 4.0 | [link](https://github.com/DenisPeskoff/2020_acl_diplomacy) |

#### Embodied and robotics (static) (2,992 requests, 17 sources)

| Task | Source | Requests | License | Link |
| --- | --- | ---: | --- | --- |
| action_safety | ASIMOV 2.0 injury scenarios: effect of a robot's instruction on the chance of injury (Jindal et al., Google DeepMind, 2025) | 167 | Not stated | [link](https://asimov-benchmark.github.io/v2/) |
| action_safety | Robot drive commands on a room floor plan (generated) | 167 | CC BY 4.0 (generated by S1-Arena) | generated by us |
| action_safety | SafeText: situations with safe and physically unsafe advice (Levy et al., EMNLP 2022) | 166 | MIT | [link](https://github.com/sharonlevy/SafeText) |
| failure_type | ACPBench, test and validation splits (Kokel et al., AAAI 2025): plan validation, multiple-choice form (failure category) and the first inapplicable action (failure step) | 250 | CDLA-Permissive-2.0 | [link](https://huggingface.co/datasets/ibm-research/acp_bench) |
| failure_type | PlanBench plan verification task, Blocksworld and Logistics (Valmeekam et al., 2023) | 250 | MIT | [link](https://github.com/karthikv792/LLMs-Planning) |
| feasibility | ACPBench, test and validation splits (Kokel et al., AAAI 2025): action applicability, yes/no and multiple-choice forms | 167 | CDLA-Permissive-2.0 | [link](https://huggingface.co/datasets/ibm-research/acp_bench) |
| feasibility | AI2-THOR object metadata (ProcTHOR's dump of 8,715 objects in 195 scenes) and placement restrictions (SimObjType.cs) | 167 | Apache-2.0 | [link](https://github.com/allenai/ai2thor) |
| feasibility | VirtualHome object properties (properties_data.json) and the action checks of its evolving-graph simulator (Puig et al., 2018) | 166 | MIT | [link](https://github.com/xavierpuigf/virtualhome) |
| next_skill | ACPBench Hard, next action towards the goal, test and validation splits (Kokel et al., 2025) | 159 | CDLA-Permissive-2.0 | [link](https://huggingface.co/datasets/ibm-research/acp_bench) |
| next_skill | ALFWorld text games built on ALFRED scenes, PDDL problems of the 0.4.2 release (Shridhar et al., ICLR 2021); answers by exhaustive search | 167 | MIT | [link](https://github.com/alfworld/alfworld) |
| next_skill | PlanBench Blocksworld problems, next action on a shortest plan (problems from Valmeekam et al., 2023; answers by exhaustive search) | 166 | MIT | [link](https://github.com/karthikv792/LLMs-Planning) |
| spatial_reasoning | LLM-BabyBench Predict: final agent pose after an action sequence in BabyAI grid worlds (2025) | 167 | MIT | [link](https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench) |
| spatial_reasoning | SpatialEval text-only split, Spatial-Map and Maze-Nav tasks (Wang et al., NeurIPS 2024) | 167 | Not stated | [link](https://huggingface.co/datasets/MilaWang/SpatialEval) |
| spatial_reasoning | StepGame, test set with distracting statements (Shi et al., AAAI 2022; regenerated February 2024) | 166 | MIT | [link](https://huggingface.co/datasets/michaelszx/StepGame) |
| task_success | ACPBench, test and validation splits (Kokel et al., AAAI 2025): plan validation, yes/no form | 167 | CDLA-Permissive-2.0 | [link](https://huggingface.co/datasets/ibm-research/acp_bench) |
| task_success | EB-ALFRED trajectory dataset of EmbodiedBench: 11 vision-language models in AI2-THOR (Yang et al., ICML 2025) | 167 | Not stated | [link](https://huggingface.co/datasets/EmbodiedBench/EB-Alfred_trajectory_dataset) |
| task_success | PlanBench Blocksworld plan generation: plans written by 13 language models with VAL checks (Valmeekam et al., 2023-2025) | 166 | MIT | [link](https://github.com/karthikv792/LLMs-Planning) |

### Closed-loop environments

Paused environments wait for each decision; real-time environments keep running while the model decides, so a slow model acts on an old state. Seeds are 0 to episodes − 1.

#### Embodied and robotics (16 environments)

| Environment | Source | Mode | Episodes | Link |
| --- | --- | --- | ---: | --- |
| household_loop/alfworld | ALFWorld (Shridhar et al., 2021), TextWorld version, out-of-distribution evaluation games (valid_unseen) | paused | 24 | [link](https://github.com/alfworld/alfworld) |
| mujoco_control/half_cheetah | Gymnasium HalfCheetah-v5 (MuJoCo) | paused | 10 | [link](https://gymnasium.farama.org/environments/mujoco/) |
| mujoco_control/hopper | Gymnasium Hopper-v5 (MuJoCo) | paused | 10 | [link](https://gymnasium.farama.org/environments/mujoco/) |
| mujoco_control/inverted_double_pendulum | Gymnasium InvertedDoublePendulum-v5 (MuJoCo) | paused | 10 | [link](https://gymnasium.farama.org/environments/mujoco/) |
| mujoco_control/inverted_pendulum | Gymnasium InvertedPendulum-v5 (MuJoCo) | paused | 10 | [link](https://gymnasium.farama.org/environments/mujoco/) |
| mujoco_control/reacher | Gymnasium Reacher-v5 (MuJoCo) | paused | 10 | [link](https://gymnasium.farama.org/environments/mujoco/) |
| tabletop_loop/button_press_topdown | Meta-World button-press-topdown-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/door_open | Meta-World door-open-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/drawer_close | Meta-World drawer-close-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/drawer_open | Meta-World drawer-open-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/peg_insert_side | Meta-World peg-insert-side-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/pick_place | Meta-World pick-place-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/push | Meta-World push-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/reach | Meta-World reach-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/window_close | Meta-World window-close-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |
| tabletop_loop/window_open | Meta-World window-open-v3 (MuJoCo) | paused | 10 | [link](https://github.com/Farama-Foundation/Metaworld) |

#### Games (57 environments)

| Environment | Source | Mode | Episodes | Link |
| --- | --- | --- | ---: | --- |
| chess_play/stockfish_level0 | Stockfish 19 at skill level 0 (the weakest official setting), move choice re-implemented with a seeded generator | paused | 20 | [link](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19) |
| open_world/crafter | Crafter (Hafner, 2022; crafter 1.8.3), default world, through BALROG's text wrapper | paused | 10 | [link](https://github.com/danijar/crafter) |
| roguelike/babyai_goto_local | BabyAI GoToLocal (BabyAI-GoToLocal-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/babyai_goto_obj | BabyAI GoToObj (BabyAI-GoToObj-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/babyai_mini_boss_level | BabyAI MiniBossLevel (BabyAI-MiniBossLevel-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/babyai_pickup_loc | BabyAI PickupLoc (BabyAI-PickupLoc-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/babyai_put_next_local | BabyAI PutNextLocal (BabyAI-PutNextLocal-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/babyai_unlock_pickup | BabyAI UnlockPickup (BabyAI-UnlockPickup-v0, Minigrid) | paused | 10 | [link](https://minigrid.farama.org/environments/babyai/) |
| roguelike/minihack_boxoban_hard | MiniHack MiniHack-Boxoban-Hard-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_boxoban_medium | MiniHack MiniHack-Boxoban-Medium-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_corridor_r3 | MiniHack MiniHack-Corridor-R3-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_corridorbattle_dark | MiniHack MiniHack-CorridorBattle-Dark-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_mazewalk_15x15 | MiniHack MiniHack-MazeWalk-15x15-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_mazewalk_9x9 | MiniHack MiniHack-MazeWalk-9x9-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_quest_easy | MiniHack MiniHack-Quest-Easy-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/minihack_quest_medium | MiniHack MiniHack-Quest-Medium-v0 (Samvelyan et al., 2021; BALROG's fork 3ecb6da on balrog-nle 0.9.0), a BALROG task | paused | 10 | [link](https://github.com/balrog-ai/minihack) |
| roguelike/nethack | NetHack 3.6.6 in the NetHack Learning Environment, NetHackChallenge-v0 (Küttler et al., 2020; balrog-nle 0.9.0), a BALROG task | paused | 5 | [link](https://github.com/balrog-ai/nle) |
| rule_puzzles/baba_is_ai/goto_win | Baba Is AI env/goto_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/goto_win-distr_obj | Baba Is AI env/goto_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/goto_win-distr_obj-irrelevant_rule | Baba Is AI env/goto_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/goto_win-distr_obj_rule | Baba Is AI env/goto_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/goto_win-distr_rule | Baba Is AI env/goto_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/make_win | Baba Is AI env/make_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/make_win-distr_obj | Baba Is AI env/make_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/make_win-distr_obj-irrelevant_rule | Baba Is AI env/make_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/make_win-distr_obj_rule | Baba Is AI env/make_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/make_win-distr_rule | Baba Is AI env/make_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-goto_win | Baba Is AI env/two_room-break_stop-goto_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-goto_win-distr_obj | Baba Is AI env/two_room-break_stop-goto_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-goto_win-distr_obj-irrelevant_rule | Baba Is AI env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-goto_win-distr_obj_rule | Baba Is AI env/two_room-break_stop-goto_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-goto_win-distr_rule | Baba Is AI env/two_room-break_stop-goto_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-make_win | Baba Is AI env/two_room-break_stop-make_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-make_win-distr_obj | Baba Is AI env/two_room-break_stop-make_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-make_win-distr_obj-irrelevant_rule | Baba Is AI env/two_room-break_stop-make_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-make_win-distr_obj_rule | Baba Is AI env/two_room-break_stop-make_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-break_stop-make_win-distr_rule | Baba Is AI env/two_room-break_stop-make_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win | Baba Is AI env/two_room-goto_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win-distr_obj | Baba Is AI env/two_room-goto_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win-distr_obj-irrelevant_rule | Baba Is AI env/two_room-goto_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win-distr_obj_rule | Baba Is AI env/two_room-goto_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win-distr_rule | Baba Is AI env/two_room-goto_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-goto_win-distr_win_rule | Baba Is AI env/two_room-goto_win-distr_win_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_wall_win | Baba Is AI env/two_room-make_wall_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win | Baba Is AI env/two_room-make_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win-distr_obj | Baba Is AI env/two_room-make_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win-distr_obj-irrelevant_rule | Baba Is AI env/two_room-make_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win-distr_obj_rule | Baba Is AI env/two_room-make_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win-distr_rule | Baba Is AI env/two_room-make_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_win-distr_win_rule | Baba Is AI env/two_room-make_win-distr_win_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_you | Baba Is AI env/two_room-make_you (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-make_you-make_win | Baba Is AI env/two_room-make_you-make_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-maybe_break_stop-goto_win | Baba Is AI env/two_room-maybe_break_stop-goto_win (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-maybe_break_stop-goto_win-distr_obj | Baba Is AI env/two_room-maybe_break_stop-goto_win-distr_obj (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule | Baba Is AI env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-maybe_break_stop-goto_win-distr_obj_rule | Baba Is AI env/two_room-maybe_break_stop-goto_win-distr_obj_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |
| rule_puzzles/baba_is_ai/two_room-maybe_break_stop-goto_win-distr_rule | Baba Is AI env/two_room-maybe_break_stop-goto_win-distr_rule (Cloos et al., 2024; baba-is-ai at 40ae1b2), a BALROG task | paused | 3 | [link](https://github.com/nacloos/baba-is-ai) |

#### Agent loops (text, web, tools, coding) (51 environments)

| Environment | Source | Mode | Episodes | Link |
| --- | --- | --- | ---: | --- |
| coding_controller/swebench_verified_gpt41mini | SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; candidate actions written by GPT-4.1 mini (temperature 0, 5 per state) through the AI Gateway | paused | 40 | [link](https://www.swebench.com/) |
| coding_controller/swebench_verified_gpt56luna | SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; candidate actions written by GPT-5.6 Luna (reasoning effort none, 5 per state) through Amazon Bedrock | paused | 40 | [link](https://www.swebench.com/) |
| coding_controller/swebench_verified_scripted | SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), official evaluation images and swebench 5.0.2 grading; candidate actions from a scripted proposer: the reference patch and 3 failed patches of public SWE-bench submissions | paused | 40 | [link](https://www.swebench.com/) |
| customer_service/tau2_telecom_solo | τ²-bench (Barres et al., 2025), telecom domain, no-user (solo) mode; tau2 1.0.1 at commit b7ea907 | paused | 114 | [link](https://github.com/sierra-research/tau2-bench) |
| science_experiments/scienceworld | ScienceWorld (Wang et al., 2022), 30 task types, test variations, 'easy' simplifications; ScienceWorld 1.3.0 at commit e8216d6 | paused | 30 | [link](https://github.com/allenai/ScienceWorld) |
| text_games/jericho_adventureland | Jericho (Hausknecht et al., 2020) game suite: Adventureland (Scott Adams, 1978; Inform port, 1996) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_deephome | Jericho (Hausknecht et al., 2020) game suite: Deephome: A Telleen Adventure (Joshua Wise, 1999) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_detective | Jericho (Hausknecht et al., 2020) game suite: Detective (Matt Barringer, ported by Stuart Moore, 2000) | paused | 1 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_dragon | Jericho (Hausknecht et al., 2020) game suite: Dragon Adventure (William Stott) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_library | Jericho (Hausknecht et al., 2020) game suite: All Quiet on the Library Front (Michael S. Phillips, 1995) | paused | 1 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_ludicorp | Jericho (Hausknecht et al., 2020) game suite: The Ludicorp Mystery (Cal Henderson, 2003) | paused | 1 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_omniquest | Jericho (Hausknecht et al., 2020) game suite: OMNIQuest (Chris Barden and Chris Ethridge, c. 1988; Inform port) | paused | 1 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_pentari | Jericho (Hausknecht et al., 2020) game suite: Pentari (Howard A. Sherman, 1998) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_zork1 | Jericho (Hausknecht et al., 2020) game suite: Zork I: The Great Underground Empire (Infocom, 1980) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/jericho_zork3 | Jericho (Hausknecht et al., 2020) game suite: Zork III: The Dungeon Master (Infocom, 1982) | paused | 3 | [link](https://github.com/microsoft/jericho) |
| text_games/tw_cooking_level3 | TextWorld cooking game (First TextWorld Problems generator), GATA test set, difficulty level 3: 1 ingredient to find and cut; 1 room | paused | 20 | [link](https://github.com/xingdi-eric-yuan/GATA-public) |
| text_games/tw_cooking_level5 | TextWorld cooking game (First TextWorld Problems generator), GATA test set, difficulty level 5: 1 ingredient to find; 9 rooms | paused | 20 | [link](https://github.com/xingdi-eric-yuan/GATA-public) |
| text_games/tw_cooking_level7 | TextWorld cooking game (First TextWorld Problems generator), GATA test set, difficulty level 7: 1 ingredient to find, cut and cook; 1 room | paused | 20 | [link](https://github.com/xingdi-eric-yuan/GATA-public) |
| text_games/tw_cooking_level9 | TextWorld cooking game (First TextWorld Problems generator), GATA test set, difficulty level 9: 3 ingredients to find, cut and cook; 6 rooms | paused | 20 | [link](https://github.com/xingdi-eric-yuan/GATA-public) |
| web/miniwob/book-flight-nodelay | MiniWoB++ book-flight-nodelay (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/buy-ticket | MiniWoB++ buy-ticket (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/choose-date-nodelay | MiniWoB++ choose-date-nodelay (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/choose-list | MiniWoB++ choose-list (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-button | MiniWoB++ click-button (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-button-sequence | MiniWoB++ click-button-sequence (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-checkboxes-large | MiniWoB++ click-checkboxes-large (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-checkboxes-soft | MiniWoB++ click-checkboxes-soft (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-collapsible-2-nodelay | MiniWoB++ click-collapsible-2-nodelay (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-dialog-2 | MiniWoB++ click-dialog-2 (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-option | MiniWoB++ click-option (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-scroll-list | MiniWoB++ click-scroll-list (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-tab-2-hard | MiniWoB++ click-tab-2-hard (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/click-widget | MiniWoB++ click-widget (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/copy-paste-2 | MiniWoB++ copy-paste-2 (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/email-inbox | MiniWoB++ email-inbox (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/email-inbox-forward-nl-turk | MiniWoB++ email-inbox-forward-nl-turk (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/enter-date | MiniWoB++ enter-date (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/enter-password | MiniWoB++ enter-password (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/form-sequence-2 | MiniWoB++ form-sequence-2 (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/login-user-popup | MiniWoB++ login-user-popup (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/multi-layouts | MiniWoB++ multi-layouts (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/navigate-tree | MiniWoB++ navigate-tree (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/phone-book | MiniWoB++ phone-book (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/read-table-2 | MiniWoB++ read-table-2 (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/search-engine | MiniWoB++ search-engine (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/social-media | MiniWoB++ social-media (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/social-media-some | MiniWoB++ social-media-some (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/tic-tac-toe | MiniWoB++ tic-tac-toe (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web/miniwob/use-autocomplete-nodelay | MiniWoB++ use-autocomplete-nodelay (Liu et al., 2018; miniwob 1.1) | paused | 5 | [link](https://miniwob.farama.org/) |
| web_shopping/webshop | WebShop (Yao et al., 2022), text mode, all 1.18M products, human instructions (test split); WebShop code at commit 64fa2a5 | paused | 100 | [link](https://github.com/princeton-nlp/WebShop) |
| wikiracing/wikispeedia | Wikispeedia (West & Leskovec, 2012): 2007 Wikipedia Selection for Schools, its link graph and finished human games | paused | 20 | [link](https://snap.stanford.edu/data/wikispeedia.html) |

#### Real-time environments (14 environments)

| Environment | Source | Mode | Episodes | Link |
| --- | --- | --- | ---: | --- |
| atari/boxing | Boxing (Atari 2600) in the Arcade Learning Environment, objects from OCAtari | realtime | 10 | [link](https://github.com/k4ntz/OC_Atari) |
| atari/breakout | Breakout (Atari 2600) in the Arcade Learning Environment, objects from OCAtari | realtime | 10 | [link](https://github.com/k4ntz/OC_Atari) |
| atari/freeway | Freeway (Atari 2600) in the Arcade Learning Environment, objects from OCAtari | realtime | 10 | [link](https://github.com/k4ntz/OC_Atari) |
| atari/pong | Pong (Atari 2600) in the Arcade Learning Environment, objects from OCAtari | realtime | 10 | [link](https://github.com/k4ntz/OC_Atari) |
| doom/basic | ViZDoom basic scenario (Kempka et al., 2016) | realtime | 10 | [link](https://vizdoom.farama.org/environments/default/) |
| doom/deadly_corridor | ViZDoom deadly_corridor scenario (Kempka et al., 2016) | realtime | 10 | [link](https://vizdoom.farama.org/environments/default/) |
| doom/defend_the_center | ViZDoom defend_the_center scenario (Kempka et al., 2016) | realtime | 10 | [link](https://vizdoom.farama.org/environments/default/) |
| doom/health_gathering | ViZDoom health_gathering scenario (Kempka et al., 2016) | realtime | 10 | [link](https://vizdoom.farama.org/environments/default/) |
| drone/quadx_waypoints | PyFlyt QuadX-Waypoints-v4 (Tai et al., 2023) | realtime | 10 | [link](https://github.com/jjshoots/PyFlyt) |
| highway/highway | highway-env highway-v0 (Leurent, 2018) | realtime | 10 | [link](https://github.com/Farama-Foundation/HighwayEnv) |
| highway/intersection | highway-env intersection-v0 (Leurent, 2018) | realtime | 10 | [link](https://github.com/Farama-Foundation/HighwayEnv) |
| highway/intersection_multi_agent | highway-env intersection-multi-agent-v0 (Leurent, 2018), destinations changed so that the paths conflict | realtime | 10 | [link](https://github.com/Farama-Foundation/HighwayEnv) |
| highway/merge | highway-env merge-v0 (Leurent, 2018) | realtime | 10 | [link](https://github.com/Farama-Foundation/HighwayEnv) |
| highway/roundabout | highway-env roundabout-v0 (Leurent, 2018) | realtime | 10 | [link](https://github.com/Farama-Foundation/HighwayEnv) |
