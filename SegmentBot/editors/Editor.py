NAME = "Editor"
DESCRIPTION = "Editor message."

class Editor():

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Editor.registry[cls.getName()] = cls

    @staticmethod
    def get_Subclasses():
        return Editor.registry

    @staticmethod
    def edit(img, mask):
        raise NotImplementedError("Interface method. Please call a subclass.")

    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION