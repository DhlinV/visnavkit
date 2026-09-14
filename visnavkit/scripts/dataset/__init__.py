"""Corpus tooling: preprocess, cache targets, fit statistics, cluster anchors, inspect samples.

Everything downstream of ``cache`` reads the cached window targets rather than the videos, so
fitting normalizer statistics, clustering anchors and plotting distributions cost one pass over
the pose sidecars instead of a decode per window.
"""
