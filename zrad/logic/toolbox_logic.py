class Image:
    def __init__(self, array, origin, spacing, direction, shape, dtype):
        self.array = array
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.shape = shape
        self.dtype = dtype
