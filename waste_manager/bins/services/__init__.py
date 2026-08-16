"""
Service layer.

Each module here is a thin adapter: it reads Django models, calls into
``wastebins_core`` for the actual computation, and writes results back.  No
algorithm is implemented in this package -- that is deliberate, so the code
which produced the published numbers is provably the code that ships.
"""
