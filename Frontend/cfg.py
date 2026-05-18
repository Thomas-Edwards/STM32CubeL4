import os

class Config:
    def __init__(self, nfilt=26, nfeat=13,
                 nfft=512, rate=16000):
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.model_path = os.path.join('Model', 'alarm.keras')
        self.p_path = os.path.join('Pickle', 'alarm.p')

