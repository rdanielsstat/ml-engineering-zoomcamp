# rebuild_pipeline.py
import pickle
from pipeline import LOSPipeline

# load your old pipeline or recreate it
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# now save it again with LOSPipeline imported from pipeline.py
with open("pipeline_v2.bin", "wb") as f_out:
    pickle.dump(pipeline, f_out)