import sys
import os
import torch
import pytest
import unittest
import pathlib

def test_PYTHONPATH():
    print(os.environ)
    print(os.environ.get('PYTHONPATH'))
    print(pathlib.Path(os.environ.get('PYTHONPATH')).absolute() == pathlib.Path("/home/r06942045/myProjects/Meta-TTS"))
    assert pathlib.Path(os.environ.get('PYTHONPATH')).absolute() == pathlib.Path("/home/r06942045/myProjects/Meta-TTS")
