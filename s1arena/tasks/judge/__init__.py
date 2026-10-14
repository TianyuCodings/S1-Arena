"""Domain: LLM-as-Judge / evaluators (评估器).

Every leaf keeps only labels that come from the source datasets (human annotations,
execution results or program checks), never preference votes. Pairwise comparisons
are asked in both orders so that position bias cancels out.
"""
