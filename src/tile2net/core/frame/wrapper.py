class Wrapper:

    @property
    def wrapper(self):
        return self

    @wrapper.setter
    def wrapper(self, value):
        ...

    @wrapper.deleter
    def wrapper(self):
        ...

