import abc
from functools import reduce
from pathlib import Path
from typing import Generator

import numpy as np

import cv2

def compile(file: str, /, *, output : str | None = None, scale: str = "fit", height: int = 10, show: bool = False, asm: bool = False):
    input_path = Path(file)
    resolution = (Instruction.WORD_SIZE - 1, height)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")
    image = cv2.imread(str(input_path.absolute()))
    image = lex_pass(image, scale, resolution)
    if show:
        cv2.imshow("image", image)
        while cv2.waitKey(0) != ord('q'):
            pass
        cv2.destroyAllWindows()
    image = parse_pass(image)
    if asm:
        if output is None:
            output = Path("a.s")
        with open(output, "w") as f:
            for instruction in image:
                f.write(str(instruction))
    else:
        if output is None:
            output = Path("a.out")
        with open(output, "wb") as f:
            for instruction in image:
                f.write(instruction.as_word())


def lex_pass(image: cv2.Mat, scale: str, resolution: tuple[int, int]):

    if image is None:
        raise FileNotFoundError("The image file was not found or cannot be opened.")

    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = ~image # invert image

    if scale == 'stretch':
        # Stretch the image to fit exactly into the target size
        image = cv2.resize(image, resolution)
    elif scale == 'fit':
        # Fit the image to the target size while maintaining aspect ratio
        original_height, original_width = image.shape
        target_width, target_height = resolution
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            # Horizontal image
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Vertical image
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))

        # Create a 640x320 canvas and place the resized image in the center
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        start_x = (target_width - new_width) // 2
        start_y = (target_height - new_height) // 2
        canvas[start_y:start_y + new_height, start_x:start_x + new_width] = image
        image = canvas
    elif scale == 'crop':
        # Crop the center of the image to fit into the target size
        original_height, original_width = image.shape
        target_width, target_height = resolution

        center_x, center_y = original_width // 2, original_height // 2
        crop_x1 = max(center_x - target_width // 2, 0)
        crop_y1 = max(center_y - target_height // 2, 0)
        crop_x2 = min(center_x + target_width // 2, original_width)
        crop_y2 = min(center_y + target_height // 2, original_height)

        image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        image = cv2.resize(image, (target_width, target_height))
    else:
        raise ValueError("Invalid scale_option. Available options: 'stretch', 'fit', 'crop'.")

    image = cv2.resize(image, resolution)

    return image

"""
The syntax divides a bread as multi-tape turing SIMD machine of push-down automata with 4 instructions
Due to the nature of python I doubt this will be performant
PUSH
POP
TOAST
JMP
"""
class Instruction(abc.ABC):
    WORD_SIZE = 11

    def __repr__(self):
        return NotImplemented
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def as_word_impl(self) -> bytes:
        pass

    def as_word(self):
        output: bytes = self.as_word_impl()

        assert len(output) <= self.WORD_SIZE
        output += b"\0" * (len(output) - self.WORD_SIZE)
        return output

class Push(Instruction):
    def as_word_impl(self) -> bytes:
        return b"\1"

    def __str__(self):
        return "PUSH"

class Noop(Instruction):
    def as_word_impl(self):
        return b"\0"

    def __str__(self):
        return "PUSH"
class Pop(Instruction):
    def as_word_impl(self):
        return b"\2"

    def __str__(self):
        return "POP"
class Toast(Instruction):
    inner: list[int]

    def __init__(self, *args: int):
        super().__init__()
        assert len(args) + 1 == self.WORD_SIZE
        self.inner = list(args)

    def __repr__(self):
        return self.inner

    def __str__(self):
        all_toasts = " ".join(str(a) for a in self.inner)
        return f"TOAST {all_toasts}"

    def as_word_impl(self) -> bytes:
        return reduce(lambda x, y : x + y, (bytes(x)[0:1] for x in self.inner), b"\3")

class Jmp(Instruction):
    inner: int
    def __init__(self, address: int):
        super().__init__()
        self.inner = address

    def __str__(self):
        return f"JMP {self.inner}"

    def as_word_impl(self) -> bytes:
        return b"\3" + b"\0" * (self.WORD_SIZE - 2) + bytes(self.inner)[0:1]

def parse_pass(mat: cv2.Mat) -> Generator[Instruction]:
    np_mat = np.array(mat)

    for row in np_mat:
        yield Toast(*row)
        yield Pop()
