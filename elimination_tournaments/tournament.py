"""
This defines a double elimination 'Tournament' object.
"""
import abc
import math
import itertools
from collections import defaultdict
from enum import Enum
from typing import List, Any, Iterator, Tuple

from elimination_tournaments.match import Match
from elimination_tournaments.participant import Participant


class StagesEnum(Enum):
    Finals = 2
    SemiFinals = 4
    QuarterFinals = 8
    Last16 = 16
    Last32 = 32
    Last64 = 64
    Last128 = 128
    ForBronze = 3
    For5th = 5
    For7th = 7
    For9th = 9
    For11th = 11
    For13th = 13
    For15th = 15
    For17th = 17
    For19th = 19
    For21th = 21
    For23th = 23
    For25th = 25
    For27th = 27
    For29th = 29
    For31th = 31


class DoubleEliminationTournament:
    """
    This is a double-elimination tournament where each match is between 2 competitors.
    When a competitor loses they are sent to the losers bracket where they'll play until
    they lose again or they make it to the final match against the winner of the winners bracket.
    It does not handle a second "grand finals" match, that should be handled outside of this object.
    It takes in a list of competitors, which can be strings or any type of Python object,
    but they should be unique. They should be ordered by a seed, with the first entry being the most
    skilled and the last being the least. They can also be randomized before creating the instance.
    Optional options dict fields:
    """
    def __init__(self, competitors_list, bracket_reset_finals=False):
        self.match_counter = itertools.count(1)
        # Only tournaments with 2 or more competitors are valid.
        assert len(competitors_list) > 1
        self.__matches = []
        self.__bracket_reset_finals = bracket_reset_finals
        # Find minimum 'n' such that 2^n >= number of competitors
        next_higher_power_of_two = int(math.pow(2, math.ceil(math.log2(len(competitors_list)))))
        # Since the bracket is fundamentally a binary tree with 2^n nodes,
        # assign the winners enough byes to fill out the 2^n slots.
        winners_number_of_byes = next_higher_power_of_two - len(competitors_list)
        # Create participants for first round (real and empty)
        incoming_participants = list(map(Participant, competitors_list))
        incoming_participants.extend([None] * winners_number_of_byes)
        # Keep track of the participants at the end of the winner's and
        # loser's brackets. Later, we will assemble these into the finals match.
        last_winner = None
        last_loser = None

        losers_by_round = []
        while len(incoming_participants) > 1:
            next_higher_power_of_two = int(math.pow(2, math.ceil(math.log2(len(incoming_participants)))))
            current_stage = StagesEnum(next_higher_power_of_two*2)
            losers = []
            # Split participants into best and worst
            half_length = int(len(incoming_participants)/2)
            first = incoming_participants[0:half_length]
            last = incoming_participants[half_length:]
            last.reverse()

            next_round_participants = []
            for participant_pair in zip(first, last):
                # If we have only one participant, send that participant
                # directly to the next round.
                if participant_pair[1] is None:
                    next_round_participants.append(participant_pair[0])
                elif participant_pair[0] is None:
                    next_round_participants.append(participant_pair[1])
                # If we have two participants, generate a match and send
                # the winner of the match to the next winner's round,
                # and the loser of the match to the loser's bracket.
                else:
                    # match = Match(participant_pair[0], participant_pair[1])
                    # self.__matches.append(match)
                    match = self._create_match(*participant_pair,
                                               stage=str(current_stage.name))
                    next_round_participants.append(match.get_winner_participant())
                    last_winner = match.get_winner_participant()
                    losers.append(match.get_loser_participant())
            # If we have any losers, create a new losers round.
            # This condition means there will be no empty loser's rounds.
            if len(losers) > 0:
                losers_by_round.append(losers)
            incoming_participants = next_round_participants

        # If we gave anybody bye's in the winner's bracket and there are
        # more than 1 loser's bracket rounds, then skip the first loser's
        # bracket round and merge it with the second loser's bracket round.
        if winners_number_of_byes > 0 and len(losers_by_round) > 1:
            losers_by_round[1].extend(losers_by_round[0])
            losers_by_round = losers_by_round[1:]

        # Mix in empty rounds to the loser's bracket. This gives extra 'room'
        # such that we can sufficiently thin out the loser's bracket
        # to match the number of incoming participants from the winner's
        # bracket in each round.
        # Rationale: For any round beyond the first, the loser's brakcet will 
        # receive 'n' participants from the previous round of the winner's
        # bracket and the loser's bracket. In the next round, we will receive
        # 'n/2' participants from the winner's bracket. Thus, we need to trim
        # down the loser's bracket by a factor of 4, which will take 2 rounds.
        empty_by_round = []
        for __ in losers_by_round:
            empty_by_round.append([])
        losers_by_round = list(itertools.chain(*zip(losers_by_round, empty_by_round)))
        # If there are more than 2 loser's bracket rounds, then
        # remove the 1st empty round from the loser's bracket and keep the rest.
        # Effectively, this disables the thinning for round 1.
        # Round 1 is a special case because it is the ONLY round of the loser's
        # bracket where we ONLY receive participants from the winner's bracket.
        if len(losers_by_round) > 2:
            new_losers = [losers_by_round[0]]
            new_losers.extend(losers_by_round[2:])
            losers_by_round = new_losers

        # Reverse participants every 4 loser's bracket rounds.
        for loser_round in range(0, len(losers_by_round), 4):
            losers_by_round[loser_round].reverse()

        # Create loser's bracket using loser participants from winner's bracket.
        index = 0
        incoming_participants = []
        for losers in losers_by_round:
            incoming_participants = losers

            if len(incoming_participants) > 1:
                # Find minimum 'n' such that 
                # 2^n < number of participants in this round.
                next_higher_power_of_two = int(math.pow(2, math.ceil(math.log2(len(incoming_participants)))))
                current_stage = StagesEnum(next_higher_power_of_two*2)
                # Since every round has a different number of matches
                # in the winners bracket ( non-trivial due to winner's byes),
                # we compute the number of bye's in the loser's bracket
                # on a per-round basis.
                number_of_byes = next_higher_power_of_two - len(incoming_participants)
                incoming_participants.extend([None] * number_of_byes)
                # Loser's bracket is also seeded so match top competitors
                # with bottom competitors
                half_length = math.ceil(len(incoming_participants)/2)
                first = incoming_participants[0:half_length]
                last = incoming_participants[half_length:]
                last.reverse()

                incoming_participants = []
                for participant_pair in zip(first, last):
                    # If we have only one participant, send that participant
                    # directly to the next round.
                    if participant_pair[0] is None:
                        incoming_participants.append(participant_pair[1])
                    elif participant_pair[1] is None:
                        incoming_participants.append(participant_pair[0])
                    else:
                        # If we have two participants, generate a match and send
                        # the winner of the match to the next loser's round,
                        # match = Match(participant_pair[0], participant_pair[1])
                        # self.__matches.append(match)
                        match = self._create_match(*participant_pair,
                                                   stage=str(current_stage.name))
                        incoming_participants.append(match.get_winner_participant())
                if len(incoming_participants) > 0:
                    # If this is the last round
                    if len(losers_by_round) <= index + 1:
                        # Create a new round.
                        losers_by_round.append(incoming_participants)
                    # Otherwise, if there is another round
                    else:
                        # Send our (outputted) participants to that round.
                        losers_by_round[index + 1].extend(incoming_participants)
            # If there are 0 or 1 participants in this round, and there is
            # a future round, send the participants there.
            elif len(losers_by_round) > index + 1:
                losers_by_round[index + 1].extend(incoming_participants)
            # If this round only has 1 participant, then set this participant
            # as the winner of the loser's bracket.
            if len(incoming_participants) == 1:
                last_loser = incoming_participants[0]
            index += 1

        # Generate finals match.
        # Important: the incoming winner should always be the first participant to determine bracket reset
        # finals_match = Match(last_winner, last_loser)
        # self.__matches.append(finals_match)
        finals_match = self._create_match(last_winner, last_loser,
                                          stage=str(StagesEnum.F.name))
        self.__finals_match = finals_match
        
        if bracket_reset_finals:
            # bracket_reset_finals_match = Match(finals_match.get_winner_participant(), finals_match.get_loser_participant())
            # self.__matches.append(bracket_reset_finals_match)
            bracket_reset_finals_match = self._create_match(finals_match.get_winner_participant(),
                                                            finals_match.get_loser_participant(),
                                                            stage=str(StagesEnum.F.name))
            # The winner of the overall tournament is the winner of the
            # bracket reset finals match.
            self.__winner = bracket_reset_finals_match.get_winner_participant()
            self.__bracket_reset_finals_match = bracket_reset_finals_match
        else:
            self.__winner = finals_match.get_winner_participant()

    def _create_match(self, left_team, right_team, stage: str = None):
        match_i = next(self.match_counter)
        new_match = Match(left_team, right_team, i=match_i, stage=stage)
        self.__matches.append(new_match)
        return new_match

    def __iter__(self):
        return iter(self.__matches)

    def __repr__(self) -> str:
        winner = self.__winner
        num_matches = len(self.__matches)
        return f'<Tournament winner={winner} num_matches={num_matches}>'

    def get_active_matches(self):
        """
        Returns a list of all matches that are ready to be played.
        """
        return [match for match in self.get_matches() if match.is_ready_to_start()]

    def get_matches(self):
        """
        Returns a list of all matches for the tournament.
        """
        return self.__matches

    def get_active_matches_for_competitor(self, competitor):
        """
        Given the string or object of the competitor that was supplied
        when creating the tournament instance,
        returns a list of Match's that they are currently playing in.
        """
        matches = []
        for match in self.get_active_matches():
            competitors = [participant.get_competitor() for participant in match.get_participants()]
            if competitor in competitors:
                matches.append(match)
        return matches

    def get_winners(self):
        """
        Returns None if the tournament is done, otherwise
        returns list of the one victor.
        """
        if len(self.get_active_matches()) > 0:
            return None
        return [self.__winner.get_competitor()]

    def add_win(self, match, competitor):
        """
        Set the victor of a match, given the competitor string/object and match.
        """
        match.set_winner(competitor)
        # If we show a match after the winner of the lower bracket beats the winner of the upper bracket
        if self.__bracket_reset_finals:
            finals = self.__finals_match
            bracket_reset = self.__bracket_reset_finals_match
            # If the finals match is played but the bracket reset match is not
            if finals.get_winner_participant().get_competitor() is not None:
                if bracket_reset.get_winner_participant().get_competitor() is None:
                    # If the incoming winner of the finals match won the finals match, then don't play the reset
                    if finals.get_winner_participant().get_competitor() is finals.get_participants()[0].get_competitor():
                        self.add_win(bracket_reset, finals.get_winner_participant().get_competitor())


class SingleEliminationTournament:
    """
    This is a single-elimination tournament where each match is between 2 competitors.
    It takes in a list of competitors, which can be strings or any type of Python object,
    but they should be unique. They should be ordered by a seed, with the first entry being the most
    skilled and the last being the least. They can also be randomized before creating the instance.
    Optional options dict fields:
    """
    def __init__(self, competitors_list, max_place=1, start_match_counter=1,
                 all_places_played=True):
        self.max_place = max_place
        self.sub_eliminations = []
        self.match_counter = itertools.count(start_match_counter)
        assert len(competitors_list) > 1
        self.__matches = []
        next_higher_power_of_two = int(math.pow(2, math.ceil(math.log2(len(competitors_list)))))
        winners_number_of_byes = next_higher_power_of_two - len(competitors_list)
        incoming_participants = []
        for competitor in competitors_list:
            if isinstance(competitor, Participant):
                incoming_participants.append(competitor)
            else:
                incoming_participants.append(Participant(competitor))
        incoming_participants.extend([None] * winners_number_of_byes)
        self.participants = incoming_participants[:]

        while len(incoming_participants) > 1:
            next_higher_power_of_two = int(math.pow(2, math.ceil(math.log2(len(incoming_participants)))))
            current_stage = StagesEnum(next_higher_power_of_two)
            # if current_stage == StagesEnum.Finals and max_place > 1:
            #     current_stage = StagesEnum(max_place)
            half_length = int(len(incoming_participants)/2)
            first = incoming_participants[0:half_length]
            last = incoming_participants[half_length:]
            last.reverse()
            next_round_participants = []
            participants_out = []
            for participant_pair in zip(first, last):
                if participant_pair[1] is None:
                    next_round_participants.append(participant_pair[0])
                elif participant_pair[0] is None:
                    next_round_participants.append(participant_pair[1])
                else:
                    match = self._create_match(*participant_pair, stage=current_stage.name)
                    next_round_participants.append(match.get_winner_participant())
                    participants_out.append(match.get_loser_participant())

            incoming_participants = next_round_participants
            if all_places_played and len(participants_out) >= 2:
                new_ko = SingleEliminationTournament(
                    participants_out,
                    max_place=int(next_higher_power_of_two/2) + max_place,
                    start_match_counter=start_match_counter + self.total_number_of_matches + len(incoming_participants) - 1
                )
                self.sub_eliminations.append(new_ko)
        self.__winner = incoming_participants[0]

    @property
    def winner(self):
        return self.__winner

    def get_own_standing(self):
        stage2matches = dict(self.own_matches_by_stage)
        final_match = stage2matches[StagesEnum.Finals]
        winner = final_match[0].get_winner_participant()
        loser = final_match[0].get_loser_participant()
        return {
            self.max_place: winner.get_competitor(),
            self.max_place+1: loser.get_competitor()
        }

    def get_complete_standing(self):
        standings = dict()
        for et in self.iter_all_eliminations():
            standings.update(et.get_own_standing())
        out = [x[1] for x in sorted(standings.items(), key=lambda z: z[0])]
        return out

    def _create_match(self, left_team, right_team, stage: str = None):
        match_i = next(self.match_counter)
        new_match = Match(left_team, right_team, i=match_i, stage=stage)
        self.__matches.append(new_match)
        return new_match

    def __iter__(self):
        return iter(self.__matches + sum([list(x) for x in self.sub_eliminations], []))

    @property
    def own_matches(self) -> Iterator[Match]:
        return iter(self.__matches)

    @property
    def own_matches_by_stage(self) -> List[Tuple[StagesEnum, List[Match]]]:
        stage2match = dict()
        for m in self.__matches:
            m: Match
            try:
                stage2match[StagesEnum[m.stage]].append(m)
            except KeyError:
                stage2match[StagesEnum[m.stage]] = [m]
        out = sorted(stage2match.items(), key=lambda x: x[0].value, reverse=True)
        return out

    @property
    def all_matches_places_by_stage(self) -> List[Tuple[StagesEnum, List[Tuple[int, List[Match]]]]]:
        stage2match = defaultdict(list)
        match2place = dict()
        for et in self.iter_all_eliminations():
            for m in et.own_matches:
                match2place[m.id] = et.max_place
                stage2match[StagesEnum[m.stage]].append(m)
        stage2places_matches = dict()
        for stage, ms in stage2match.items():
            place2matches = defaultdict(list)
            for m in ms:
                place2matches[match2place[m.id]].append(m)
            stage2places_matches[stage] = place2matches.items()
        out = sorted(stage2places_matches.items(), key=lambda x: x[0].value, reverse=True)
        return out

    def add_win_by_competitor(self, competitor: Any):
        self.add_win(self.get_active_matches_for_competitor(competitor)[0], competitor)

    def iter_all_eliminations(self):
        yield self
        for se in self.sub_eliminations:
            for x in se.iter_all_eliminations():
                yield x

    @property
    def total_number_of_matches(self):
        num_matches = len(list(self.__iter__()))
        return num_matches

    def __repr__(self) -> str:
        winner = self.__winner
        return f'<Tournament winner={winner} num_matches={self.total_number_of_matches} max_place={self.max_place}>'

    def get_unfinished_matches(self):
        return [match for match in self if match.is_unfinished()]

    def get_own_unfinished_matches(self):
        return [match for match in self.own_matches if match.is_unfinished()]

    def pre_populate_matches(self, losing_participant: Participant):
        out = []
        for i, match in enumerate(self.__matches):
            l_p, r_p = match.get_participants()
            if str(l_p) == str(losing_participant):
                winning_participant = r_p
            elif str(r_p) == str(losing_participant):
                winning_participant = l_p
            else:
                continue
            match.set_winner(str(winning_participant),
                             winner_resolved=winning_participant.resolved,
                             loser_resolved=losing_participant.resolved)
            out.append(self.__matches.pop(i))
            break
        return out

    def get_active_matches(self):
        """
        Returns a list of all matches that are ready to be played.
        """
        return [match for match in self if match.is_ready_to_start()]

    def get_matches(self):
        """
        Returns a list of all matches for the tournament.
        """
        return self.__matches

    def get_active_matches_for_competitor(self, competitor):
        """
        Given the string or object of the competitor that was supplied
        when creating the tournament instance,
        returns a list of Match's that they are currently playing in.
        """
        matches = []
        for match in self.get_active_matches():
            competitors = [participant.get_competitor() for participant in match.get_participants()]
            if competitor in competitors:
                matches.append(match)
        return matches

    def get_winners(self):
        """
        Returns None if the tournament is done, otherwise
        returns list of the one victor.
        """
        if len(self.get_active_matches()) > 0:
            return None
        return [self.__winner.get_competitor()]

    def add_win(self, match, competitor):
        """
        Set the victor of a match, given the competitor string/object and match.
        """
        match.set_winner(competitor)