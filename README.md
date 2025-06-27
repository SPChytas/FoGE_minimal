# FoGE_minimal


The main functionality is provided in the `utils` folder.

1. Create you own Vector Symbolic Architecture in `vsa.py`. Each VSA is repsented as a class which inherits the abstract class VSA.

2. Each attribute of a dataset (e.g., name of node) is represented as an object of a class in `attr_info.py`. Supported types of attributes so far are: categorical, vector (including single-dimensional continuous), and text attributes. Each new type/class of attribute should adhere to the abstract class Attr.

3. After obtaining the attribute information (a collection of objects from above), we create the vectors/symbols associated with each attribute. For categorical attributes we create a vector per different value, for textual features we employ a text encoder, and for continuous/vector attributes we use a random projection to the new space. Supported vector/symbol creators are: text encoder based, orthonormal, gaussian/random. The corresponding functions can be found on `vectors_creators.py`.

4. After obtain the vocabulary (i.e., the symbol/vector associated with each value of each attribute), we employ an encoder to encode the graphical structures. The encoder is located in `encoders.py` and it encodes the graph's structure, all node attributes and all edge attributes according to the formulas of the paper.


Examples of the whole encoding pipeline can be found on dataset_preparation.py