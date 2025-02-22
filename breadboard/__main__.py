import fire
from pathlib import Path
from breadboard.compiler import compile

def run(file: Path):
    print(file)

if __name__ == '__main__':
    fire.Fire(name="breadboard")

