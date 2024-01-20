# Simple Temperature Cool-down in Contrastive Framework for Unsupervised Sentence Representation Learning
----
This repo contains implementation of Simple Temperature Cool-down, released to facilitate research.

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

# Overview

We proposes a simple, tricky method to improve sentence representation of unsupervised contrastive learning. Temperature cool-down technique helps PLMs to be more suitable for contrastive learning via preparation of uniform representation space. PCA visualization of the representation space during contrastive learning with/without temperature cool-down. (a): Following the literature, BERT-base shows the anisotropic representation space. (b): A model trained with temperature cool-down pulls distant instances (colored pink) more uniformly. (c): A representation space built by temperature cool-down leads to a more uniform unit hypersphere.
