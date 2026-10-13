"""Text-form measurements for the proficiency-cue validation.

Pure-Python where possible (edit distance, MATTR, MTLD, repetition, SUBTLEX
lookup); spaCy for lemmas, POS and dependency syntax; optional external
instruments (LanguageTool, a CoLA acceptability model) behind small clients.
Every instrument reports its own version so the frozen config can pin it.
"""
