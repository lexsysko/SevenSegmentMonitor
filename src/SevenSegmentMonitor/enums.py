from enum import StrEnum, auto


class MaskType(StrEnum):
    WITHOUT_GREEN = auto()
    RED = auto()
    RED_ADAPTIVE = auto()
