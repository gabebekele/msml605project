from pathlib import Path
from scripts.ingest_lfw import load


def main():
    ds, info = load()
    
    print("Dataset loaded!")
    print("Dataset info:")
    print(info)
    
    # Print number of examples
    num_examples = info.splits['train'].num_examples
    print(f"Number of examples in train split: {num_examples}")

    # Look at 5 examples
    print("\nFirst 5 examples:")
    for i, example in enumerate(ds.take(5)):
        print(example)
        if i >= 4:
            break

if __name__ == "__main__":
    main()