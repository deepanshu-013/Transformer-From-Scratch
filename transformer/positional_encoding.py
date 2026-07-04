import numpy as np

class PositionalEncoding:
    @staticmethod
    def forward(sequence_length, dimension):

        positional_encoding = np.zeros((sequence_length, dimension), dtype="float32")

        for pos in range(sequence_length):
            for dim in range(dimension):
                if dim % 2 == 0:
                    positional_encoding[pos, dim] = np.sin(pos/10000**(dim/dimension))
                else:
                    positional_encoding[pos, dim] = np.cos(pos/10000**((dim - 1)/dimension))

        return positional_encoding






















