from pathlib import Path


"""
This script containes utility functions for the project.
"""


def get_paths() -> tuple:
    """
    Get the paths to the repository, data, and model directories.

    Returns:
        tuple: (repo_path, data_path, model_path)
    """

    repo_path = Path(__file__).resolve().parents[1]
    data_path = repo_path / "data"
    model_path = repo_path / "models"

    return repo_path, data_path, model_path

def run_profiling():
    """
    Run profiling on the project.
    """

    from torch.profiler import profile, ProfilerActivity
    from models.model import TextClassificationModel
    # Get model
    model = TextClassificationModel(
        model_name="distilbert-base-uncased",
        num_labels=2,
        learning_rate=0.01,
    )
    # Get the tokenizer
    tokenizer = model.tokenizer
    # Dummy data, e.g. 5 sentences
    texts = ["This is a test"] * 5
    #labels = [0] * 5
    # Tokenize
    inputs = tokenizer(texts, return_tensors="pt")
    # Forward pass
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        outputs = model(**inputs)
        #logits = outputs.logits

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("outputs/profiling/trace.json")

    """
    Interpretation of results:
    aten::addmm is the most time-consuming operation, 59%
    - this is the matrix multiplication, the self-attention mechanism
    - the goal of this mechanism is to learn the relationships between words
    """

    # Now using cProfile
    import cProfile
    import pstats
    from io import StringIO
    pr = cProfile.Profile()
    pr.enable()
    outputs = model(**inputs)
    #logits = outputs.logits
    pr.disable()
    s = StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    """
    Interpretation of results:
    ??
    """


if __name__ == "__main__":
    print(get_paths())


    # profiling
    run_profiling()