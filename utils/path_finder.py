import os

dirname = os.path.dirname(__file__) #current directory


NILMTK_SOURCE = os.path.join(dirname, "../../Datasets")
SOURCES = os.path.join(dirname, "../data/data")
SCENARIOS = os.path.join(dirname, "../data/scenarios")
PRETRAINED = os.path.join(dirname, "../experiments/pretrained_models")
RESULTS = os.path.join(dirname, "../results")
