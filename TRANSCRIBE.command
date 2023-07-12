#!/bin/bash
cd "${0%/*}"
pipenv sync --python 3.10
pipenv run python3 AudioTranscriber_v1.0_executable.py 

