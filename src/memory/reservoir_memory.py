import random

class ReservoirMemory:

    def __init__(self, max_size:int=300_000):
        self.buffer = []
        self.max_size = max_size
        self.length = 0
        
    def add(self, sample):
        if self.length < self.max_size:
            self.buffer.append(sample)
            self.length += 1
        else:
            i = random.randint(0, self.length - 1)
            self.buffer[i] = sample
            
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

    def sample_all(self):
        return self.buffer
    
    