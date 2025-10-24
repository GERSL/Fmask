import numpy as np
np.seterr(invalid='ignore') # ignore the invalid errors

class BitLayer:
    """
    Represents a collection of boolean layers stored as a 2D array of bits.
    Each layer is represented by a single bit in the array.
    """

    def __init__(self, shape):
        self.index = None # Initialize the index of the current layer
        self.shape = shape
        self.layers = np.zeros(shape, dtype=np.uint8)  # Initialize a 2D array with uint8 (8-bit integers)

    def append(self, mask, value = True):
        """
        Append the boolean value of a specific layer at given positions.
        :param layer: Layer index (0 to 7)
        :param mask: 2D boolean array indicating positions to set
        :param value: Boolean value (True or False)
        """
        if self.index is None: # If no layer has been appended yet
            self.index = 0
        else: # If a layer has been appended, move to next layer
            self.index += 1
        self.set(self.index, mask, value = value)

    def set(self, layer, mask, value = True):
        """
        Set the boolean value of a specific layer at given positions.
        :param layer: Layer index (0 to 7)
        :param mask: 2D boolean array indicating positions to set
        :param value: Boolean value (True or False)
        """
        if layer < 0 or layer > 7:
            raise ValueError("Layer index must be between 0 and 7.")
        if value:
            self.layers[mask] |= (1 << layer)  # Set the bit to 1
        else:
            self.layers[mask] &= ~(1 << layer)  # Set the bit to 0

    @property
    def length(self):
        """
        Returns the length of the object.
        
        The length is calculated as the current index plus one.
        
        Returns:
            int: The length of the object.
        """
        return self.index + 1
    @property
    def first(self):
        """
        Get the boolean values of the first layer for all positions.
        :return: 2D boolean array with values of the first layer
        """
        return (self.layers & 1) != 0

    @property
    def last(self):
        """
        Get the boolean values of the last layer for all positions.
        :return: 2D boolean array with values of the last layer
        """
        return (self.layers & (1 << self.index)) != 0

    def get(self, layer):
        """
        Get the boolean values of a specific layer for all positions.
        :param layer: Layer index (0 to 7)
        :return: 2D boolean array with values of the specified layer
        """
        if layer < 0 or layer > 7:
            raise ValueError("Layer index must be between 0 and 7.")
        return (self.layers & (1 << layer)) != 0

    def toggle(self, layer, mask):
        """
        Toggle the boolean value of a specific layer at given positions.
        :param layer: Layer index (0 to 7)
        :param mask: 2D boolean array indicating positions to toggle
        """
        if layer < 0 or layer > 7:
            raise ValueError("Layer index must be between 0 and 7.")
        self.layers[mask] ^= (1 << layer)

    def __repr__(self):
        """
        String representation of the array for debugging purposes.
        """
        return np.array2string(self.layers, formatter={'int': lambda x: f'{x:08b}'})

# # Example usage
# height, width = 5, 5
# bit_layer_manager = BitLayer2DManager(height, width)

# # Create masks for setting layers
# mask1 = np.array([[True, False, False, False, False],
#                   [False, True, False, False, False],
#                   [False, False, False, False, False],
#                   [False, False, False, False, False],
#                   [False, False, False, False, False]])

# mask2 = np.array([[False, False, False, False, False],
#                   [False, False, True, False, False],
#                   [False, False, False, False, False],
#                   [False, False, False, True, False],
#                   [False, False, False, False, False]])

# # Set some layers using masks
# bit_layer_manager.set_layer(0, mask1, True)
# bit_layer_manager.set_layer(1, mask2, True)

# # Get layer values
# print(bit_layer_manager.get_layer(0))  # 2D boolean array with values of layer 0
# print(bit_layer_manager.get_layer(1))  # 2D boolean array with values of layer 1

# # Toggle a layer using a mask
# bit_layer_manager.toggle_layer(0, mask1)  # Toggle layer 0 at positions specified by mask1

# # Print the bit representation of the 2D array
# print(bit_layer_manager)
