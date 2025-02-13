#!/usr/bin/env bash
set -e -v
black QuantumPioneer
ruff check --fix QuantumPioneer
