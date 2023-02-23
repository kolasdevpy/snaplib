class Counter:
    def __init__(self, start=0):
        self.counter = start

    def __call__(self, step=1):
        self.counter += step
        return self.counter

