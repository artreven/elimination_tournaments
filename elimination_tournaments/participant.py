"""
The Participant class represents a participant in a specific match.
It can be used as a placeholder until the participant is decided.
"""

class Participant:
    """
    The Participant class represents a participant in a specific match.
    It can be used as a placeholder until the participant is decided.
    """
    def __init__(self, competitor: str, resolved=True):
        self.competitor = competitor
        self.resolved = resolved

    def __repr__(self) -> str:
        if self.resolved:
            return f'<Participant {self.competitor}>'
        else:
            return f'[{self.competitor}]'

    def get_competitor(self):
        """
        Return the competitor that was set,
        or None if it hasn't been decided yet
        """
        return self.competitor

    def set_competitor(self, competitor):
        """
        Set competitor after you've decided who it will be,
        after a previous match is completed.
        """
        self.competitor = competitor
