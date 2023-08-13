"""
A match represents a single match in a tournament, between 2 participants.
It adds empty participants as placeholders for the winner and loser,
so they can be accessed as individual object pointers.
"""
from typing import List

from elimination_tournaments.participant import Participant


class Match:
    """
    A match represents a single match in a tournament, between 2 participants.
    It adds empty participants as placeholders for the winner and loser,
    so they can be accessed as individual object pointers.
    """
    def __init__(self, left_participant, right_participant,
                 i: int = None,
                 stage: str = None):
        self.__left_participant = left_participant
        self.__right_participant = right_participant
        self.id = i
        self.__winner = Participant(f'Winner {self.id}', resolved=False)
        self.__loser = Participant(f'Loser {self.id}', resolved=False)
        self.stage = stage

    @property
    def left_participant(self) -> Participant:
        return self.__left_participant

    @left_participant.setter
    def left_participant(self, value: Participant):
        self.__left_participant = value

    @property
    def right_participant(self) -> Participant:
        return self.__right_participant

    @right_participant.setter
    def right_participant(self, value: Participant):
        self.__right_participant = value

    def __repr__(self) -> str:
        left = self.__left_participant
        right = self.__right_participant
        winner = self.__winner
        loser = self.__loser
        out = f'<Match {self.id if self.id is not None else ""} '
        if self.stage is not None:
            out += f'{self.stage} '
        out += f'left={left} right={right} winner={winner} loser={loser}> '
        return out

    def set_winner(self, competitor: Participant, loser_resolved=True, winner_resolved=True):
        """
        When the match is over, set the winner competitor here and the loser will be set too.
        """
        if str(competitor) == str(self.__left_participant):
            self.__winner.set_competitor(self.__left_participant.get_competitor())
            self.__loser.set_competitor(self.__right_participant.get_competitor())
        elif str(competitor) == str(self.__right_participant):
            self.__winner.set_competitor(self.__right_participant.get_competitor())
            self.__loser.set_competitor(self.__left_participant.get_competitor())
        else:
            raise Exception("Invalid competitor")
        self.__loser.resolved = loser_resolved
        self.__winner.resolved = winner_resolved

    def get_winner_participant(self):
        """
        If the winner is set, get it here. Otherwise this return None.
        """
        return self.__winner

    def get_loser_participant(self):
        """
        If the winner is set, you can get the loser here. Otherwise this return None.
        """
        return self.__loser

    def get_participants(self) -> List[Participant]:
        """
        Get the left and right participants in a list.
        """
        return [self.__left_participant, self.__right_participant]

    def is_unfinished(self):
        return not self.__winner.resolved

    def is_ready_to_start(self):
        """
        This returns True if both of the participants coming in have their competitors "resolved".
        This means that the match that the participant is coming from is finished.
        It also ensure that the winner hasn't been set yet.
        """
        is_left_resolved = self.__left_participant.resolved
        is_right_resolved = self.__right_participant.resolved
        is_winner_resolved = self.__winner.resolved
        return is_left_resolved and is_right_resolved and not is_winner_resolved
