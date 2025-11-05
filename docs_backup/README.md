# Documentation Index

This directory contains all documentation for the Mutation-Based Training Energy Profiler.

## Quick Start

- **[Usage Examples](USAGE_EXAMPLES.md)** - Practical examples and command-line usage
- **[Quick Reference](quick_reference.md)** - Fast reference guide for hyperparameters
- **[Configuration Explanation](CONFIG_EXPLANATION.md)** - Understanding models_config.json
- **[Config File Feature](CONFIG_FILE_FEATURE.md)** - Batch experiment configuration files

## Core Documentation

### Configuration & Setup
- **[CONFIG_EXPLANATION.md](CONFIG_EXPLANATION.md)** - Explains why models_config.json is needed and how it works
- **[CONFIG_FILE_FEATURE.md](CONFIG_FILE_FEATURE.md)** - Complete guide to batch experiment configuration files

### Usage Guides
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Detailed usage scenarios with command examples
- **[quick_reference.md](quick_reference.md)** - Quick reference for hyperparameter mutations

## Reference Documentation

### Hyperparameter Analysis
- **[hyperparameter_analysis.md](hyperparameter_analysis.md)** - Comprehensive analysis of all hyperparameters (MAIN REFERENCE)
- **[hyperparameter_support_matrix.md](hyperparameter_support_matrix.md)** - Matrix showing which hyperparameters are supported by which models
- **[original_hyperparameter_defaults.md](original_hyperparameter_defaults.md)** - Default values from original repositories

### Code Modification Guides
- **[code_modification_patterns.md](code_modification_patterns.md)** - Patterns for modifying training scripts
- **[code_modifications_log.md](code_modifications_log.md)** - Log of all code modifications made
- **[stage2_3_modification_guide.md](stage2_3_modification_guide.md)** - Guide for Stage 2 & 3 modifications

## Archived Documentation

The following files are older versions or superseded by newer documentation. They are kept for reference but may contain outdated information:

### Archived Hyperparameter Analysis
- **[archived/hyperparameters_analysis.md](archived/hyperparameters_analysis.md)** - Older version (2025-11-03), superseded by hyperparameter_analysis.md
- **[archived/hyperparameter_mutation_analysis.md](archived/hyperparameter_mutation_analysis.md)** - Merged into hyperparameter_analysis.md
- **[archived/hyperparameter_feasibility.md](archived/hyperparameter_feasibility.md)** - Feasibility analysis, merged into main analysis
- **[archived/current_hyperparameter_support_matrix.md](archived/current_hyperparameter_support_matrix.md)** - Superseded by hyperparameter_support_matrix.md
- **[archived/hyperparameter_matrix_final.md](archived/hyperparameter_matrix_final.md)** - Superseded by hyperparameter_support_matrix.md

### Archived Feature-Specific Documentation
- **[archived/precision_options_analysis.md](archived/precision_options_analysis.md)** - Analysis of fp16/bf16 options
- **[archived/precision_options.md](archived/precision_options.md)** - Detailed precision options guide
- **[archived/seed_verification_report.md](archived/seed_verification_report.md)** - Seed parameter verification
- **[archived/weight_decay_progress_report.md](archived/weight_decay_progress_report.md)** - Weight decay implementation progress
- **[archived/weight_decay_verification_report.md](archived/weight_decay_verification_report.md)** - Weight decay verification
- **[archived/hyperparameters_explained.md](archived/hyperparameters_explained.md)** - Detailed hyperparameter explanations

## Document Classification

### By Purpose
- **Setup & Configuration**: CONFIG_EXPLANATION.md, CONFIG_FILE_FEATURE.md
- **Usage & Examples**: USAGE_EXAMPLES.md, quick_reference.md
- **Analysis & Reference**: hyperparameter_analysis.md, hyperparameter_support_matrix.md, original_hyperparameter_defaults.md
- **Development**: code_modification_patterns.md, code_modifications_log.md, stage2_3_modification_guide.md

### By Audience
- **End Users**: USAGE_EXAMPLES.md, quick_reference.md, CONFIG_FILE_FEATURE.md
- **Researchers**: hyperparameter_analysis.md, hyperparameter_support_matrix.md
- **Developers**: code_modification_patterns.md, code_modifications_log.md, CONFIG_EXPLANATION.md

## File Naming Conventions

All documentation files follow these conventions:
- **UPPERCASE_WITH_UNDERSCORES.md** - Major documentation files (e.g., CONFIG_EXPLANATION.md, USAGE_EXAMPLES.md)
- **lowercase_with_underscores.md** - Reference and analysis files (e.g., hyperparameter_analysis.md, quick_reference.md)
- **archived/** subdirectory - Outdated or superseded documentation

## Contributing

When adding new documentation:
1. Use clear, descriptive filenames
2. Follow the naming conventions above
3. Add an entry to this README.md
4. Include creation/update date in the document header
5. Move superseded documents to archived/ directory
