# Commit Risk Prediction using Machine Learning

This project predicts whether a Pull Request (PR) will fail CI before running the pipeline.

## Features
- Commit message length
- Files changed
- Insertions
- Deletions

## Model
- Random Forest Classifier

## Workflow
1. Extract commit data using PyDriller
2. Generate features
3. Train ML model
4. Evaluate accuracy
