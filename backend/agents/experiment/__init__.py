"""Three-phase stoppage prediction experiment.

Phases:
  1. Train  — fit classical models on normal-only data from training operations.
  2. Test   — evaluate with frozen model + neutral priors (no feedback).
  3. Eval   — evaluate with feedback mechanic active; measure its impact.
"""
