/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GAMESTATE_H_INCLUDED
#define GAMESTATE_H_INCLUDED

#include <vector>
#include <memory>
#include <string>
#include <bitset>
#include <utility>
#include <deque>

#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "Network.h"
#include "TimeControl.h"

class GameState : public KoState {
public:
    using InputPlane = std::pair<Network::BoardPlane, Network::BoardPlane>;

    explicit GameState() = default;
    explicit GameState(const KoState* rhs) {
        // Copy in fields from base class.
        *(static_cast<KoState*>(this)) = *rhs;
        anchor_game_history();
    }
    void init_game(int size, float komi);
    void reset_game();
    bool set_fixed_handicap(int stones);
    int set_fixed_handicap_2(int stones);
    void place_free_handicap(int stones);
    void anchor_game_history(void);

    void rewind(void); /* undo infinite */
    bool undo_move(void);
    bool forward_move(void);

    void play_move(int color, int vertex);
    void play_move(int vertex);
    void play_pass();
    bool play_textmove(std::string color, std::string vertex);

    void start_clock(int color);
    void stop_clock(int color);
    TimeControl& get_timecontrol();
    void set_timecontrol(int maintime, int byotime, int byostones,
                         int byoperiods);
    void set_timecontrol(TimeControl tmc);
    void adjust_time(int color, int time, int stones);

    void display_state();

    // TODO private?
    const InputPlane& get_boardplanes(int moves_ago) const;
    void state_to_board_plane(Network::BoardPlane& our_plane, Network::BoardPlane& their_plane) const;

    void disable_history();

private:
    bool valid_handicap(int stones);
    void update_boardplanes();
    void append_to_gamehistory();

    std::vector<std::shared_ptr<KoState>> game_history;
    std::deque<InputPlane> m_boardplanes;

    TimeControl m_timecontrol;
    bool m_history_enabled = true;
};

#endif
