class PenaltyEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.blue_penalties = []
        self.red_penalties = []

    def add_blue(self, penalty):
        if penalty not in self.blue_penalties:
            self.blue_penalties.append(penalty)

    def add_red(self, penalty):
        if penalty not in self.red_penalties:
            self.red_penalties.append(penalty)

    def clear_frame(self):
        self.blue_penalties.clear()
        self.red_penalties.clear()