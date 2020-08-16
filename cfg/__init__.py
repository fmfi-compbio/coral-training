from . import tinyB as tinyB

configs = {
    "tinyB-relu6-bn90": lambda: tinyB.get_config(activation="relu6", dropout=0.05, bn_momentum=0.90),
    "tinyB-relu6-bn93": lambda: tinyB.get_config(activation="relu6", dropout=0.05, bn_momentum=0.93),
    "tinyB-relu6-bn96": lambda: tinyB.get_config(activation="relu6", dropout=0.05, bn_momentum=0.96),
    "tinyB-relu6-bn99": lambda: tinyB.get_config(activation="relu6", dropout=0.05, bn_momentum=0.99),


    "tinyB-relu6-drop00": lambda: tinyB.get_config(activation="relu6", dropout=0.0, bn_momentum=0.90),
    "tinyB-relu6-drop05": lambda: tinyB.get_config(activation="relu6", dropout=0.05, bn_momentum=0.90),
    "tinyB-relu6-drop10": lambda: tinyB.get_config(activation="relu6", dropout=0.10, bn_momentum=0.90),
    "tinyB-relu6-drop20": lambda: tinyB.get_config(activation="relu6", dropout=0.20, bn_momentum=0.90),
    "tinyB-relu6-drop30": lambda: tinyB.get_config(activation="relu6", dropout=0.30, bn_momentum=0.90),
    "tinyB-relu6-drop40": lambda: tinyB.get_config(activation="relu6", dropout=0.40, bn_momentum=0.90),
    "tinyB-relu6-drop50": lambda: tinyB.get_config(activation="relu6", dropout=0.50, bn_momentum=0.90),
}
