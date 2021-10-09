#include "World.h"

#include <stdlib.h>
#include <string.h>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <sstream>
#include "Connection.h"
#include "Replay.h"
#include "Debug.h"
#include "Inventory.h"
#include "Monster.h"
#include "Actions/Action.h"
#include "Actions/Move.h"
#include "Analyzers/Analyzer.h"
#include "Data/Item.h"
#include "Data/Skill.h"
#include "Data/Monster.h"
#include "Events/Event.h"

using namespace analyzer;
using namespace std;

/* static variables */
vector<Point> World::_changes;
char World::_view[ROWS][COLS + 1] = {
	{'\0'}
};
int World::_color[ROWS][COLS] = {
	{0}
};
Point World::_cursor;
int World::_cur_page = -1;
int World::_max_page = -1;
int World::_frame_count = 0;
bool World::_menu = false;
bool World::_question = false;
char World::_levelname[MAX_LEVELNAME_LENGTH] = {'\0'};
int World::_turn = 0;
int World::_sub_turn = -1;
int World::_min_saiph_energy = 12;
int World::_max_saiph_energy = 12;
unsigned int World::_internal_turn = 0;
vector<Level> World::_levels;

Connection* World::_connection = NULL;
action::Action* World::_action = NULL;
action::Action* World::_last_action = NULL;
list<action::Action*> World::_action_queue;
bool World::_changed[MAP_ROW_END + 1][MAP_COL_END + 1] = {
	{false}
};
string World::_messages = " ";
bool World::_inverse = false;
bool World::_bold = false;
char World::_data[BUFFER_SIZE * 2] = {'\0'};
int World::_data_size = -1;
string World::_msg_str;
string World::_msg_buffer = "";
Point World::_last_menu;
map<string, vector<int> > World::_levelmap;
timeval World::_start_time;
vector<Analyzer*> World::_analyzers;
int World::_last_action_id = NO_ACTION;
Coordinate World::_branches[BRANCHES] = {Coordinate()};
Point World::_cout_cursor;
int World::_cout_last_color;
unsigned char World::_shadow_map_dump[MAP_ROW_END - MAP_ROW_BEGIN + 1][MAP_COL_END - MAP_COL_BEGIN + 1][2];
std::string World::_shadow_rhs[50];

static struct termios _save_termios, _current_termios;
static int _current_speed;
int World::_display_level = -1;
map<char, pair<void*, World::drawfunc> > World::_draw_funcs;
map<char, pair<void*, World::drawfunc> >::iterator World::_current_draw_func;

const char* World::_ansi_colors[] = {
	/*   0 */ "0", "0;1", "", "", "", "", "", "0;7", "", "",
	/*  10 */ "", "", "", "", "", "", "", "", "", "",
	/*  20 */ "", "", "", "", "", "", "", "", "", "",
	/*  30 */ "0;30", "0;31", "0;32", "0;33", "0;34", "0;35", "0;36", "0;37", "", "",
	/*  40 */ "0;7;30", "0;7;31", "0;7;32", "0;7;33", "0;7;34", "0;7;35", "0;7;36", "0;7;37", "", "",
	/*  50 */ "", "", "", "", "", "", "", "", "", "",
	/*  60 */ "", "", "", "", "", "", "", "", "", "",
	/*  70 */ "", "", "", "", "", "", "", "", "", "",
	/*  80 */ "", "", "", "", "", "", "", "", "", "",
	/*  90 */ "0;1;30", "0;1;31", "0;1;32", "0;1;33", "0;1;34", "0;1;35", "0;1;36", "0;1;37", "", "",
	/* 100 */ "1;7;30", "1;7;31", "1;7;32", "1;7;33", "1;7;34", "1;7;35", "1;7;36", "1;7;37",
};


char _text[ROWS*COLS + 1];

/* methods */
char World::view(const Point& point) {
	return _view[point.row()][point.col()];
}

int World::color(const Point& point) {
	return _color[point.row()][point.col()];
}

const Point& World::cursor() {
	return _cursor;
}

bool World::menu() {
	return _menu;
}

bool World::question() {
	return _question;
}

int World::curPage() {
	return _cur_page;
}

int World::maxPage() {
	return _max_page;
}

int World::turn() {
	return _turn;
}

unsigned int World::internalTurn() {
	return _internal_turn;
}

int World::subTurn() {
	return _sub_turn;
}

const vector<Point>& World::changes() {
	return _changes;
}

const Coordinate& World::branchCoordinate(int branch) {
	return _branches[branch];
}

Level& World::level() {
	return level(Saiph::position().level());
}

Level& World::level(int level) {
	return _levels[level];
}

int World::findLevel(int branch, int depth) {
	for (unsigned i = 0; i < _levels.size(); ++i) {
		if (_levels[i].branch() == branch && _levels[i].depth() == depth)
			return i;
	}
	return -1;
}

const vector<Level>& World::levels() {
	return _levels;
}

int World::currentPriority() {
	if (_action == NULL)
		return action::Action::NOOP.priority();
	return _action->command().priority();
}

action::Action* World::lastAction() {
	return _last_action;
}

std::string World::lastData() {
	// return string(_data, _data_size);
	// return string(_text, ROWS*COLS);
	return string(_text[_cursor.row() * COLS], (ROWS-_cursor.row())*COLS);
}

int World::lastActionID() {
	/* return the id of the last action */
	return _last_action_id;
}

void World::init(const string& logfile, int connection_type) {
	Debug::init(logfile);
	data::Monster::init();
	data::Item::init();
	data::Skill::init();
	Level::init();
	Analyzer::init();

	registerDrawFunc('n', &drawNormal, NULL);
	registerDrawFunc('c', &drawCosts, NULL);
	registerDrawFunc('d', &drawDirections, NULL);
	registerDrawFunc('l', &drawLight, NULL);
	_current_draw_func = _draw_funcs.find('n');

	_connection = Connection::create(connection_type);
	if (_connection == NULL) {
		cout << "ERROR: Don't know what interface this is: " << connection_type << endl;
		destroy();
		exit(1);
	}

	/* set start time */
	gettimeofday(&World::_start_time, NULL);

	cout << "\033[2J";

	/* fetch the first "frame" */
	update();
}

void World::destroy() {
	Debug::notice() << "Quitting gracefully" << endl;
	Analyzer::destroy();
	Level::destroy();
	data::Monster::destroy();
	data::Item::destroy();
	Debug::destroy();

	delete _action;
	delete _connection;
}

void World::registerAnalyzer(Analyzer* const analyzer) {
	Debug::info() << "Registering analyzer " << analyzer->name() << endl;
	_analyzers.push_back(analyzer);
}

void World::registerDrawFunc(char key, drawfunc fnc, void* cookie) {
	_draw_funcs[key] = make_pair(cookie, fnc);
}

void World::unregisterAnalyzer(Analyzer* const analyzer) {
	Debug::info() << "Unregistering analyzer " << analyzer->name() << endl;
	for (vector<Analyzer*>::iterator a = _analyzers.begin(); a != _analyzers.end(); ++a) {
		if ((*a)->name() == analyzer->name()) {
			_analyzers.erase(a);
			return;
		}
	}
}

bool World::setAction(action::Action* action, bool deleteAction) {
	if (World::_action != NULL) {
		if (action->command().priority() <= World::_action->command().priority()) {
			if (deleteAction)
				delete action;
			return false; // already got an action with higher priority
		}
		if (World::_action != World::_last_action)
			delete World::_action;
	}
	World::_action = action;
	return true;
}

bool World::queueAction(action::Action* action) {
	if (action == NULL) {
		return false; // shouldn't happen, though
	} else if (action->command().priority() <= PRIORITY_TURN_MAX) {
		/* not a zero-turn action, can't queue it */
		delete action;
		return false;
	}
	_action_queue.push_back(action);
	return true;
}

unsigned char World::directLine(Point point, bool ignore_sinks, bool ignore_boulders, int eff_range, int danger_range) {
	int dist = Point::gridDistance(Saiph::position(), point);

	/* outside map */
	if (point.row() < MAP_ROW_BEGIN || point.row() > MAP_ROW_END || point.col() < MAP_COL_BEGIN || point.col() > MAP_COL_END)
		return ILLEGAL_DIRECTION;
	if (point.row() != Saiph::position().row() && abs(point.row() - Saiph::position().row()) != dist)
		return ILLEGAL_DIRECTION;
	if (point.col() != Saiph::position().col() && abs(point.col() - Saiph::position().col()) != dist)
		return ILLEGAL_DIRECTION;
	if (dist > eff_range)
		return ILLEGAL_DIRECTION;

	static unsigned char dirs[] = { NW, N, NE, W, NOWHERE, E, SW, S, SE };
	int index = 4;

	if (point.row() < Saiph::position().row())
		index -= 3;
	else if (point.row() > Saiph::position().row())
		index += 3;

	if (point.col() < Saiph::position().col())
		index -= 1;
	else if (point.col() > Saiph::position().col())
		index += 1;

	unsigned char dir = dirs[index];
	int realized_range = 0;
	bool ok = false;
	Point temp = Saiph::position();

	// Follow out the beam as far as it can go.  If it reaches our target and can't reach a friendly, fire!
	while (true) {
		if (temp.row() < MAP_ROW_BEGIN || temp.row() > MAP_ROW_END || temp.col() < MAP_COL_BEGIN || temp.col() > MAP_COL_END)
			break;
		const Tile& t = level().tile(temp);
		if (!Level::isPassable(t.symbol()) && (!ignore_boulders || t.symbol() != BOULDER))
			break;
		if (temp == point)
			ok = true;
		const map<Point, Monster*>::const_iterator m = level().monsters().find(temp);
		if (t.monster() != ILLEGAL_MONSTER && m != level().monsters().end() && m->second->visible() && (realized_range < dist || m->second->attitude() == FRIENDLY))
			return ILLEGAL_DIRECTION;
		if (realized_range == danger_range || (!ignore_sinks && t.symbol() == SINK))
			break;
		++realized_range;
		temp.moveDirection(dir);
	}

	return ok ? dir : ILLEGAL_DIRECTION;
}

std::string World::cursorMoves(Point source, const Point& target) {
	std::string cursor_moves = "";
	while (source != target) {
		if (source.row() < target.row() && source.col() < target.col()) {
			source.moveSoutheast();
			cursor_moves.push_back(SE);
		} else if (source.row() < target.row() && source.col() > target.col()) {
			source.moveSouthwest();
			cursor_moves.push_back(SW);
		} else if (source.row() > target.row() && source.col() < target.col()) {
			source.moveNortheast();
			cursor_moves.push_back(NE);
		} else if (source.row() > target.row() && source.col() > target.col()) {
			source.moveNorthwest();
			cursor_moves.push_back(NW);
		} else if (source.row() < target.row()) {
			source.moveSouth();
			cursor_moves.push_back(S);
		} else if (source.row() > target.row()) {
			source.moveNorth();
			cursor_moves.push_back(N);
		} else if (source.col() < target.col()) {
			source.moveEast();
			cursor_moves.push_back(E);
		} else {
			source.moveWest();
			cursor_moves.push_back(W);
		}
	}
	return cursor_moves;
}

Tile& World::shortestPath(const Point& point) {
	/* returns Tile for shortest path from player to target */
	return level().tile(point);
}

Tile World::shortestPath(const Coordinate& target) {
	/* returns a "fake" Tile with data for shortest path from player to target */
	if (target.level() < 0 || target.level() >= (int) _levels.size())
		return Tile(); // outside map
	else if (target.level() == Saiph::position().level())
		return _levels[Saiph::position().level()].tile(target); // target on same level

	int pivot = -1;
	int level_count = 1;
	int level_queue[_levels.size()];
	level_queue[0] = Saiph::position().level();
	bool level_added[_levels.size()];
	for (int a = 0; a < (int) _levels.size(); ++a)
		level_added[a] = false;
	level_added[Saiph::position().level()] = true;
	Tile level_tile[_levels.size()];
	level_tile[Saiph::position().level()] = shortestPath(Saiph::position());
	//Debug::pathing() << "Trying to find path to " << target << endl;
	while (++pivot < level_count) {
		/* check if target is on level */
		if (level_queue[pivot] == target.level()) {
			Tile& tile = _levels[level_queue[pivot]].tile(target);
			if (tile.cost() == UNREACHABLE)
				continue;
			else if (tile.cost() == UNPASSABLE && tile.distance() > 1)
				continue;
			/* gotta modify this tile a bit since it's on another level */
			level_tile[level_queue[pivot]].updatePath(level_tile[level_queue[pivot]].direction(), tile.distance() + level_tile[level_queue[pivot]].distance() + 1, tile.cost() + level_tile[level_queue[pivot]].cost() + 1);
			Debug::pathing() << "Found " << target << " " << level_tile[level_queue[pivot]].distance() << " tiles away, first checkpoint is " << level_tile[level_queue[pivot]].coordinate() << endl;
			return level_tile[level_queue[pivot]];
		}
		/* path to upstairs on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_UP).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_UP).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where these stairs lead
			else if (level_added[s->second])
				continue; // already added this level
			else if (Saiph::encumbrance() >= STRESSED) {
				Debug::pathing() << "can't follow upstairs because encumbrance" << endl;
				continue;
			}
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			//Debug::pathing() << "Following upstairs on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* we know where these stairs lead, add the level to the queue */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to upstairs on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(UP);
			} else {
				/* pathing to upstairs on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
		/* path to downstairs on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_DOWN).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_DOWN).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where these stairs lead
			else if (level_added[s->second])
				continue; // already added this level
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			//Debug::pathing() << "Following downstairs on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* we know where these stairs lead, add the level to the queue */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to downstairs on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(DOWN);
			} else {
				/* pathing to downstairs on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
		/* path to portals on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) MAGIC_PORTAL).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) MAGIC_PORTAL).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where this portal leads
			else if (level_added[s->second])
				continue; // already added this level
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			//Debug::info() << "Following magic portal on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* we know where this portal leads, add the level to the queue */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to portal on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(SIT);
			} else {
				/* pathing to portal on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
	}
	return Tile(); // symbol not found
}

Tile World::shortestPath(unsigned char symbol) {
	/* returns Tile for shortest path from player to nearest symbol */
	int pivot = -1;
	int level_count = 1;
	Tile best_tile;
	int level_queue[_levels.size()];
	level_queue[0] = Saiph::position().level();
	bool level_added[_levels.size()];
	for (int a = 0; a < (int) _levels.size(); ++a)
		level_added[a] = false;
	level_added[Saiph::position().level()] = true;
	Tile level_tile[_levels.size()];
	level_tile[Saiph::position().level()] = shortestPath(Saiph::position());
	Debug::pathing() << "Trying to find path to nearest '" << symbol << "'" << endl;
	while (++pivot < level_count) {
		/* path to symbols on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols(symbol).begin(); s != _levels[level_queue[pivot]].symbols(symbol).end(); ++s) {
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() == UNREACHABLE)
				continue;
			else if (tile.cost() == UNPASSABLE && tile.distance() > 1)
				continue;
			else if (tile.cost() + level_tile[level_queue[pivot]].cost() + 1 >= best_tile.cost())
				continue;
			/* this symbol is closer than the previously found one */
			if (pivot != 0) {
				/* symbol is on another level, gotta modify this tile a bit */
				best_tile = level_tile[level_queue[pivot]];
				best_tile.updatePath(best_tile.direction(), tile.distance() + best_tile.distance() + 1, tile.cost() + best_tile.cost() + 1);
			} else {
				best_tile = tile;
			}
			Debug::pathing() << "Found '" << symbol << "' at " << Coordinate(level_queue[pivot], s->first) << ", " << tile.distance() << " tiles away, first checkpoint is " << tile.coordinate() << endl;
		}
		/* path to upstairs on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_UP).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_UP).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where these stairs lead
			if (level_added[s->second])
				continue; // already added this level
			else if (Saiph::encumbrance() >= STRESSED) {
				Debug::pathing() << "can't follow upstairs because encumbrance" << endl;
				continue;
			}
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			else if (tile.cost() + level_tile[level_queue[pivot]].cost() + 1 >= best_tile.cost())
				continue;
			Debug::pathing() << "Following upstairs on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* distance to these stairs is shorter than shortest path found so far.
			 * we should check the level these stairs lead to as well */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to upstairs on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(UP);
			} else {
				/* pathing to upstairs on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
		/* path to downstairs on level */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_DOWN).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) STAIRS_DOWN).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where these stairs lead
			if (level_added[s->second])
				continue; // already added this level
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			else if (tile.cost() + level_tile[level_queue[pivot]].cost() + 1 >= best_tile.cost())
				continue;
			Debug::pathing() << "Following downstairs on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* distance to these stairs is shorter than shortest path found so far.
			 * we should check the level these stairs lead to as well */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to downstairs on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(DOWN);
			} else {
				/* pathing to downstairs on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
		/* path to levels through magic portals */
		for (map<Point, int>::const_iterator s = _levels[level_queue[pivot]].symbols((unsigned char) MAGIC_PORTAL).begin(); s != _levels[level_queue[pivot]].symbols((unsigned char) MAGIC_PORTAL).end(); ++s) {
			if (s->second == UNKNOWN_SYMBOL_VALUE)
				continue; // we don't know where this magic portal leads
			if (level_added[s->second])
				continue; // already added this level
			Tile& tile = _levels[level_queue[pivot]].tile(s->first);
			if (tile.cost() >= UNPASSABLE)
				continue;
			else if (tile.cost() + level_tile[level_queue[pivot]].cost() >= best_tile.cost())
				continue;
			Debug::info() << "Following magic portal on " << tile.coordinate() << " (" << _levels[level_queue[pivot]].name() << ") leading to level " << s->second << " (" << _levels[s->second].name() << ")" << endl;
			/* distance to this portal is shorter than shortest path found so far.
			 * we should check the level this portal leads to as well */
			level_added[s->second] = true;
			level_queue[level_count++] = s->second;
			if (pivot == 0) {
				/* pathing to downstairs on level we're standing on */
				level_tile[s->second] = tile;
				if (tile.direction() == NOWHERE)
					level_tile[s->second].direction(NOWHERE);
			} else {
				/* pathing to downstairs on another level */
				level_tile[s->second] = level_tile[level_queue[pivot]];
				level_tile[s->second].updatePath(level_tile[s->second].direction(), level_tile[s->second].distance() + tile.distance() + 1, level_tile[s->second].cost() + tile.cost() + 1);
			}
		}
	}
	return best_tile;
}

void World::endTermios() {
	tcsetattr(0, TCSANOW, &_save_termios);
}

void World::initTermios() {
	tcgetattr(0, &_save_termios);
	atexit(&World::endTermios);
	_current_termios = _save_termios;
	_current_termios.c_iflag &= ~IXON;
	_current_termios.c_lflag &= ~(ISIG | ICANON | ECHO | IEXTEN);
	_current_termios.c_cc[VTIME] = 0;
}

void World::setKeyWait(bool wait) {
	_current_termios.c_cc[VMIN] = wait ? 1 : 0;
	tcsetattr(0, TCSANOW, &_current_termios);
}

void World::doCommands() {
	char cmd;

	do {
		if (read(0, &cmd, 1) <= 0)
			return;
		switch (cmd) {
		case '0':
			_current_speed = SPEED_PAUSE;
			break;

		case '1':
			_current_speed = SPEED_SLOW;
			break;

		case '2':
			_current_speed = SPEED_FAST;
			break;

		case 'd':
			setKeyWait(true);
			if (read(0, &cmd, 1) <= 0)
				break;
			if (_display_level < 0)
				_display_level = Saiph::position().level();
			_current_draw_func = _draw_funcs.find(cmd);
			refresh();
			break;

		case '<':
			if (_display_level < 0)
				_display_level = Saiph::position().level();
			if (_display_level) {
				--_display_level;
				refresh();
			}
			break;

		case '>':
			if (_display_level < 0)
				_display_level = Saiph::position().level();
			if (_display_level < int(_levels.size()) - 1) {
				++_display_level;
				refresh();
			}
			break;

		case 12:
			refresh();
			break;

		case 'q':
		case 'Q':
			executeCommand(string(1, (char) 27));
			executeCommand(QUIT);
			executeCommand(string(1, YES));
			destroy();
			exit(0);
		}
		setKeyWait(_current_speed == SPEED_PAUSE);
	} while (_current_speed == SPEED_PAUSE);
}

void World::run(int speed) {
	int last_turn = 0;
	int stuck_counter = 0;
	int last_action_turn = -1;
	int last_action_time = 0;
	_current_speed = speed;
	initTermios();
	setKeyWait(speed == SPEED_PAUSE);
	while (true) {
		/* check if we're in the middle of an action.
		 * Inventory and Level may send events that analyzers react on and set an action,
		 * so we check if we're in the middle of an action here and remember it for later.
		 * analyzer->actionCompleted() may too set an action, so we need to call this
		 * method a bit later as well. */
		bool analyze_and_parse = (_action == NULL || _action->command() == action::Action::NOOP);
		Analyzer* prev_analyzer = (_action == NULL ? NULL : _action->analyzer());

		/* write messages to log */
		Debug::custom("Messages") << "'" << _messages << "'" << endl;

		/* let Saiph, Inventory and current level parse messages */
		Saiph::parseMessages(_messages);
		Inventory::parseMessages(_messages);
		level().parseMessages(_messages);

		if (!_question && !_menu && analyze_and_parse) {
			/* we are at the beginning of a new NetHack action.  Pay attention to the timing */
			if (last_action_turn >= 0 && last_action_time != TIME_NONE) {
				/* moreover, we remember last turn, and we did an action that takes time */
				if (last_action_time == TIME_STANDARD && (_turn - last_action_turn) > 1) {
					// if we go an entire turn without actions, assume we were paralyzed
					// note that this means move tracking won't work well if Burdened
					// not that kiting is very useful when Burdened anyway...
					// paralysis always wears off at the beginning of a turn, but we don't know energy
					last_action_time = TIME_NOMUL;
				}

				switch (last_action_time) {
					default:
						Debug::error() << "Incorrect time value " << last_action_time << endl;
						// fall through
					case TIME_MULTIPLE:
						/* last action was an extended move-oriented action.  We now know nothing about the state of monster movement */
						_sub_turn = -1;
						_min_saiph_energy = 12;
						_max_saiph_energy = 11 + Saiph::maxSpeed();
						break;
					case TIME_NOMUL:
						// our last action was something that uses full turns, or we were paralyzed.  punt on analysis for now
						_sub_turn = 0;
						_min_saiph_energy = 12;
						_max_saiph_energy = 11 + Saiph::maxSpeed();
						break;
					case TIME_STANDARD:
						/* only remaining case is one-move actions, and we sure didn't get paralyzed */
						_min_saiph_energy -= 12;
						_max_saiph_energy -= 12;
						if (_turn == last_action_turn) {
							// we didn't get any extra energy, so we still have at least 12 from the same charge
							_min_saiph_energy = std::max(12, _min_saiph_energy);
							if (_sub_turn >= 0)
								_sub_turn++;
						} else {
							_sub_turn = 0;
							// obviously we ran out of energy, so we didn't have more than 11
							_max_saiph_energy = std::min(11, _max_saiph_energy);
							if (_min_saiph_energy <= _max_saiph_energy) {
								// and a fresh charge, but don't hide contradictions 
								_min_saiph_energy += Saiph::minSpeed();
								_max_saiph_energy += Saiph::maxSpeed();
								// we have at least 12 *now*, if we didn't then we'd have seen it as a paralysis
								_min_saiph_energy = std::max(12, _min_saiph_energy);
							}
						}
						if (_min_saiph_energy > _max_saiph_energy) {
							Debug::error() << "Energy tracking for saiph has reached a contradiction.  Resetting." << endl;
							_sub_turn = -1;
							_min_saiph_energy = 12;
							_max_saiph_energy = 11 + Saiph::maxSpeed();
						}
						break;
				}
				int min_moves_turn = (_sub_turn >= 0 ? _sub_turn : 0) + (_min_saiph_energy / 12);
				Debug::info() << "Energy predictions for the current turn: t=" << _turn << " i=" << _internal_turn << " s=" << _sub_turn << " min=" << _min_saiph_energy << " max=" << _max_saiph_energy << " min_moves=" << min_moves_turn << endl;
				Saiph::minMovesThisTurn(min_moves_turn);
			}
			last_action_turn = _turn;
		}

		/* let Saiph, Inventory and current level analyze */
		if (!_question && !_menu) {
			Saiph::analyze();
			Inventory::analyze();
			level().analyze();

			/* set branch coordinate if we've discovered a new branch */
			if (_branches[level().branch()].level() == -1)
				_branches[level().branch()] = Saiph::position();

			/* dump maps */
			_display_level = -1;
			dumpMaps();
		}
		if (_current_speed == SPEED_SLOW)
			usleep(200000);
		doCommands();

		/* append this turn's messages to the buffer.
		 * if we're in a multi-message action, the messages will be queued for actionCompleted */
		_msg_buffer += _messages;

		/* analyze and parse messages if we're not continuing an action */
		if (analyze_and_parse) {
			/* let the previous analyzer know that the action was completed */
			if (prev_analyzer != NULL)
				prev_analyzer->actionCompleted(_msg_buffer);
			_msg_buffer = "";

			/* let analyzers parse messages */
			for (vector<Analyzer*>::iterator a = _analyzers.begin(); a != _analyzers.end(); ++a)
				(*a)->parseMessages(_messages);

			/* let analyzers analyze */
			if (!_question && !_menu) {
				for (vector<Analyzer*>::iterator a = _analyzers.begin(); a != _analyzers.end(); ++a)
					(*a)->analyze();
				for (vector<Analyzer*>::iterator a = _analyzers.begin(); a != _analyzers.end(); ++a)
					(*a)->lastChance(_action);
			}

			/* check if we got some queued actions */
			for (list<action::Action*>::iterator a = _action_queue.begin(); a != _action_queue.end(); ++a) {
				if (setAction(*a, false)) {
					/* we will execute this action, remove it from queue.
					 * if it fails, the analyzer that queued the action needs to handle it */
					_action_queue.erase(a);
					break;
				}
			}
		}

		/* check if we got a command */
		if (_action == NULL || _action->command() == action::Action::NOOP) {
			/* we do not. print debugging and just answer something sensible */
			if (_question) {
				Debug::warning() << "Unhandled question: " << _messages << endl;
				_last_action_id = NO_ACTION;
				executeCommand(string(1, (char) 27));
				continue;
			} else if (_menu) {
				Debug::warning() << "Unhandled menu: " << _messages << endl;
				_last_action_id = NO_ACTION;
				executeCommand(string(1, (char) 27));
				continue;
			} else {
				Debug::warning() << "I have no idea what to do... Searching 16 times" << endl;
				cout << (unsigned char) 27 << "[1;82H";
				cout << (unsigned char) 27 << "[K"; // erase everything to the right
				cout << "No idea what to do: 16s";
				/* return cursor back to where it was */
				cout << (unsigned char) 27 << "[" << _cursor.row() + 1 << ";" << _cursor.col() + 1 << "H";
				cout.flush();
				++World::_internal_turn; // will cost a turn
				_last_action_id = NO_ACTION;
				last_action_time = TIME_MULTIPLE;
				executeCommand("16s");
				continue;
			}
		}

		if (analyze_and_parse && !_question && !_menu) {
			last_action_time = _action->timeTaken();
		}

		/* print what we're doing */
		cout << (unsigned char) 27 << "[1;82H";
		cout << (unsigned char) 27 << "[K"; // erase everything to the right
		string activity = _action->analyzer()->name();
		activity += ' ';
		activity += _action->command().command();
		size_t nl_index;
		while ((nl_index = activity.find('\n')) != string::npos)
			activity.replace(nl_index, 1, "\\n");
		if (activity.size() > 78) {
			activity.erase(activity.begin() + 75, activity.end());
			activity += "...";
		}
		cout << activity;
		/* return cursor back to where it was */
		cout << (unsigned char) 27 << "[" << _cursor.row() + 1 << ";" << _cursor.col() + 1 << "H";
		/* and flush cout. if we don't do this our output looks like garbage */
		cout.flush();
		Debug::command() << "Analyzer " << _action->analyzer()->name() << " " << _action->command() << endl;

		/* execute the command */
		_last_action_id = _action->id();
		if (_last_action != _action)
			delete _last_action;
		_last_action = _action;
		executeCommand(_action->command().command());

		/* check if we're stuck */
		if (stuck_counter % 9 == 8 && _action->command().command().size() == 1) {
			Debug::warning() << "Command failed for analyzer " << _action->analyzer()->name() << ": " << _action->command() << endl;
			_action->failed();
		} else if (stuck_counter > 360) {
			/* failed too many times, #quit */
			Debug::error() << "Appear to be stuck, quitting game" << endl;
			_last_action_id = NO_ACTION;
			executeCommand(string(1, (char) 27));
			executeCommand(QUIT);
			executeCommand(string(1, YES));
			return;
		}
		if (last_turn == _turn)
			stuck_counter++;
		else
			stuck_counter = 0;
		last_turn = _turn;

		/* update current action */
		_action->update(_messages);

		/* and increase _internal_turn if the action actually cost a turn.
		 * the turn counter in game may not increase if we're [very] fast,
		 * but we still need to know if a turn lapsed, hence _internal_turn */
		if (_action->command() == action::Action::NOOP && _action->timeTaken() != TIME_NONE)
			++World::_internal_turn;
	}
}

/* private methods */
void World::refresh() {
	cout << "\033[2J";
	_cout_last_color = -1;
	_cout_cursor.row(-1);
	for (int row = 0; row < ROWS; ++row) {
		coutGoto(1 + row, 1);
		for (int col = 0; col < COLS; ++col)
			coutOneChar(_color[row][col], _view[row][col]);
	}
	memset(_shadow_map_dump, 0, sizeof _shadow_map_dump);
	for (int row = 0; row < 50; ++row)
		_shadow_rhs[row].clear();
	coutSetColor(NO_COLOR);
	dumpMaps();
	coutGoto(_cursor.row() + 1, _cursor.col() + 1);
	cout.flush();
}

void World::addChangedLocation(const Point& point) {
	coutGoto(1 + point.row(), 1 + point.col());
	coutOneChar(_color[point.row()][point.col()], _view[point.row()][point.col()]);

	/* add a location changed since last frame unless it's already added */
	if (point.row() < MAP_ROW_BEGIN || point.row() > MAP_ROW_END || point.col() < MAP_COL_BEGIN || point.col() > MAP_COL_END || _changed[point.row()][point.col()])
		return;
	_changes.push_back(point);
}

void World::detectPosition() {
	if (Saiph::position().level() < 0) {
		/* this happens when we start */
		Saiph::position(Coordinate(_levels.size(), _cursor));
		_levels.push_back(Level(_levels.size(), _levelname));
		_levelmap[_levelname].push_back(Saiph::position().level());
		return;
	}
	if ((int) _levels.size() > Saiph::position().level() && _levelname == level().name()) {
		/* same level as last frame, update row & col */
		Saiph::position(Coordinate(Saiph::position().level(), _cursor));
		return;
	}

	/* level has changed.
	 * we need to figure out if it's a new level or one we already know of */
	int found = UNKNOWN_SYMBOL_VALUE;
	unsigned char symbol = level().tile().symbol();
	/* XXX - maybe we already know where these stairs lead? */
	// there used to be code here to take advantage of such prior knowledge, but coupling level linking and level identification in that way caused horrible failures in some cases
	// for instance, stepping off a stair onto a levelporter

	// Quest levels are in one-to-one correspondance with names, and are often sufficiently odd as to confuse the wall matcher
	if (std::string(_levelname).find("Home") != string::npos) {
		vector<int> &levels = _levelmap[_levelname];
		if (levels.size())
			found = levels[0];
	}
	if (found == UNKNOWN_SYMBOL_VALUE) {
		for (vector<int>::iterator lm = _levelmap[_levelname].begin(); lm != _levelmap[_levelname].end(); ++lm) {
			/* check if level got walls on same locations.
			 * since walls can disappear, we'll allow a 80% match */
			int total = 0;
			int matched = 0;
			for (map<Point, int>::const_iterator s = _levels[*lm].symbols((unsigned char) VERTICAL_WALL).begin(); s != _levels[*lm].symbols((unsigned char) VERTICAL_WALL).end(); ++s) {
				if (_view[s->first.row()][s->first.col()] == VERTICAL_WALL)
					++matched;
				++total;
			}
			for (map<Point, int>::const_iterator s = _levels[*lm].symbols((unsigned char) HORIZONTAL_WALL).begin(); s != _levels[*lm].symbols((unsigned char) HORIZONTAL_WALL).end(); ++s) {
				if (_view[s->first.row()][s->first.col()] == HORIZONTAL_WALL)
					++matched;
				++total;
			}
			if (matched > 0 && min(matched, total) * 5 >= max(matched, total) * 4) {
				found = *lm;
				Debug::notice() << "Recognized level " << found << ": '" << _levelname << "' - '" << _levels[found].name() << "'" << endl;
				break;
			}
		}
	}
	if (found == UNKNOWN_SYMBOL_VALUE) {
		/* new level */
		found = _levels.size();
		/* when we discover a new level it's highly likely it's in the
		 * same branch as the previous level.
		 * exceptions:
		 * rogue level, next level most likely is in the main branch
		 * minetown, next level most likely is in the mines */
		int branch;
		if (level().branch() == BRANCH_ROGUE)
			branch = BRANCH_MAIN;
		else if (level().branch() == BRANCH_MINETOWN)
			branch = BRANCH_MINES;
		else
			branch = level().branch();
		_levels.push_back(Level(_levels.size(), _levelname, branch));
		_levelmap[_levelname].push_back(found);
		Debug::notice() << "Found new level " << found << ": " << _levelname << endl;
	}

	/* magic portals are much more fiddly because we usually aren't standing still when we find them */
	if (_messages.find("  You activated a magic portal!  ") != string::npos) {
		level(found).setDungeonSymbol(_cursor, MAGIC_PORTAL);
		level(found).setDungeonSymbolValue(_cursor, Saiph::position().level());
	}

	/* were we on stairs on last Saiph::position()? */
	if (symbol == STAIRS_DOWN || symbol == STAIRS_UP) {
		/* yes, we were on stairs, set where it leads */
		level().setDungeonSymbolValue(Saiph::position(), found);
		/* pretend we're standing on STAIRS_UP or STAIRS_DOWN when we discover new levels.
		 * if we're wrong, it'll fix itself */
		level(found).setDungeonSymbol(_cursor, (symbol == STAIRS_UP ? STAIRS_DOWN : STAIRS_UP));
		/* set where it leads */
		level(found).setDungeonSymbolValue(_cursor, Saiph::position().level());
	}

	/* set new position for saiph */
	level().leftLevel();
	Saiph::position(Coordinate(found, _cursor));
}

void World::coutSetColor(int color) {
	if (color != _cout_last_color) {
		cout << "\033[" << _ansi_colors[color] << 'm';
		_cout_last_color = color;
	}
}

void World::coutOneChar(int color, unsigned char ch) {
	//cout << "%c" << color << ',' << ch << '%';
	coutSetColor(color);
	cout << (unsigned char)((ch >= ' ' && ch <= '~') ? ch : '?');
	_cout_cursor.moveEast();
}

void World::coutGoto(int row, int col) {
	//cout << "%g" << row << ',' << col << '%';
	if (row != _cout_cursor.row() || col != _cout_cursor.col()) {
		cout << "\033[" << row << ';' << col << 'H';
		_cout_cursor.row(row);
		_cout_cursor.col(col);
	}
}

void World::coutRhsLine(int row, const std::string& line) {
	if (line != _shadow_rhs[row]) {
		cout << "\033[" << (1+row) << ";82H" << line << "\033[K";
		_shadow_rhs[row] = line;
	}
}

void World::drawNormal(void*, Level& lv, const Point& p, unsigned char& symbol, unsigned char& color) {
	const Tile& t = lv.tile(p);

	if ((&lv == &level()) && p.row() == Saiph::position().row() && p.col() == Saiph::position().col()) {
		color = BOLD_MAGENTA;
		symbol = '@';
	} else if (t.monster() != ILLEGAL_MONSTER) {
		color = t.monster() == _view[p.row()][p.col()] ? BOLD_RED : BOLD_YELLOW;
		symbol = t.monster();
	} else if (t.symbol() > 31 && t.symbol() < 127) {
		color = NO_COLOR;
		symbol = t.symbol();
	} else {
		color = BOLD_CYAN;
		symbol = '?';
	}
}

void World::drawLight(void*, Level& lv, const Point& p, unsigned char& symbol, unsigned char& color) {
	const Tile& t = lv.tile(p);

	if ((&lv == &level()) && p.row() == Saiph::position().row() && p.col() == Saiph::position().col()) {
		symbol = '@';
	} else if (t.monster() != ILLEGAL_MONSTER) {
		symbol = t.monster();
	} else if (t.symbol() > 31 && t.symbol() < 127) {
		symbol = t.symbol();
	} else {
		symbol = '?';
	}

	if (t.lit() == 0)
		color = RED;
	else if (t.lit() > 0)
		color = BOLD_YELLOW;
	else
		color = NO_COLOR;
}

void World::drawCosts(void*, Level& lv, const Point& p, unsigned char& symbol, unsigned char& color) {
	Tile& t = lv.tile(p);
	color = NO_COLOR;
	if ((&lv == &level()) && p.row() == Saiph::position().row() && p.col() == Saiph::position().col()) {
		color = MAGENTA;
		symbol = '@';
	} else if (t.direction() != ILLEGAL_DIRECTION) {
		unsigned cost = t.cost();
		if (cost < 140) {
			color = BOLD_RED + (cost / 20);
			symbol = 'a' + int(cost % 20);
		} else if (cost < 1540) {
			color = RED + ((cost - 140) / 200);
			symbol = 'a' + int(((cost - 140) / 10) % 20);
		} else if (cost < UNPASSABLE) {
			color = WHITE;
			symbol = 't';
		} else if (cost == UNPASSABLE) {
			color = WHITE;
			symbol = 'u';
		} else {
			color = WHITE;
			symbol = 'U';
		}
	} else
		symbol = t.symbol();
}

void World::drawDirections(void*, Level& lv, const Point& p, unsigned char& symbol, unsigned char& color) {
	Tile& t = lv.tile(p);
	color = NO_COLOR;
	if ((&lv == &level()) && p.row() == Saiph::position().row() && p.col() == Saiph::position().col()) {
		color = MAGENTA;
		symbol = '@';
	} else if (t.direction() != ILLEGAL_DIRECTION)
		symbol = t.direction();
	else
		symbol = t.symbol();
}

void World::dumpMap(Level& lv) {
	/* monsters and map as saiph sees it */
	Point p;
	_cout_last_color = -1;
	_cout_cursor.row(-1);
	if (_current_draw_func == _draw_funcs.end())
		_current_draw_func = _draw_funcs.find('n');

	void* cookie = _current_draw_func->second.first;
	drawfunc df = _current_draw_func->second.second;
	unsigned char symbol, color;

	for (p.row(MAP_ROW_BEGIN); p.row() <= MAP_ROW_END; p.moveSouth()) {
		for (p.col(MAP_COL_BEGIN); p.col() <= MAP_COL_END; p.moveEast()) {
			df(cookie, lv, p, symbol, color);

			unsigned char* old = &_shadow_map_dump[p.row() - MAP_ROW_BEGIN][p.col() - MAP_COL_BEGIN][0];
			if (symbol != old[0] || color != old[1]) {
				coutGoto(p.row() + 26, p.col() + 1);
				coutOneChar(color, symbol);
				old[0] = symbol;
				old[1] = color;
			}
		}
	}
}

void World::dumpMaps() {
	/* XXX: World echoes output from the game in the top left corner */
	/* frames/turns per second */
	timeval cur_time;
	gettimeofday(&cur_time, NULL);
	double seconds = (cur_time.tv_sec + cur_time.tv_usec / 1000000.0) - (_start_time.tv_sec + _start_time.tv_usec / 1000000.0);
	if (seconds == 0.0)
		seconds = 0.001;
	double fps = (double) _frame_count / seconds;
	double tps = (double) _turn / seconds;
	cout << (unsigned char) 27 << "[25;1H";
	if (_display_level == -1) {
		cout << "Frames/second: " << (unsigned char) 27 << "[32m";
		cout << (int) fps << "." << (int) (fps * 10) % 10;
		cout << (unsigned char) 27 << "[0m     ";
		cout << (unsigned char) 27 << "[25;26H";
		cout << "Turns/second: " << (unsigned char) 27 << "[32m";
		cout << (int) tps << "." << (int) (tps * 10) % 10;
		cout << (unsigned char) 27 << "[0m     ";
		cout << (unsigned char) 27 << "[25;51H";
		cout << "Run time: " << (unsigned char) 27 << "[32m";
		cout << (int) seconds << "." << (int) (seconds * 10) % 10;
		cout << (unsigned char) 27 << "[0m     ";
	} else {
		ostringstream status;
		status << "Currently displaying level id " << _display_level << " at depth " << level(_display_level).depth() << ", branch " << level(_display_level).branch();
		cout << "\033[25;1H" << status.str() << string(' ', 80 - status.str().size());
	}

	dumpMap(_display_level < 0 ? level() : level(_display_level));

	/* status & inventory */
	cout << "\033[m";
	std::string buf;
	if (Saiph::intrinsics() & PROPERTY_COLD)
		buf += "\033[1;34mCold ";
	if (Saiph::intrinsics() & PROPERTY_DISINT)
		buf += "\033[1;35mDisInt ";
	if (Saiph::intrinsics() & PROPERTY_FIRE)
		buf += "\033[1;31mFire ";
	if (Saiph::intrinsics() & PROPERTY_POISON)
		buf += "\033[1;32mPoison ";
	if (Saiph::intrinsics() & PROPERTY_SHOCK)
		buf += "\033[1;36mShock ";
	if (Saiph::intrinsics() & PROPERTY_SLEEP)
		buf += "\033[1;33mSleep ";
	coutRhsLine(1, buf);

	buf.clear();
	if (Saiph::intrinsics() & PROPERTY_ESP)
		buf += "\033[1;35mESP ";
	if (Saiph::intrinsics() & PROPERTY_SPEED)
		buf += "\033[1;31mSpeed ";
	if (Saiph::intrinsics() & PROPERTY_STEALTH)
		buf += "\033[1;34mStealth ";
	if (Saiph::intrinsics() & PROPERTY_TELEPORT_CONTROL)
		buf += "\033[1;36mTeleCon ";
	if (Saiph::intrinsics() & PROPERTY_TELEPORT)
		buf += "\033[1;33mTeleport ";
	if (Saiph::intrinsics() & PROPERTY_LYCANTHROPY)
		buf += "\033[1;31mLycan ";
	if (Saiph::hurtLeg())
		buf += "\033[1;34mLeg ";
	if (Saiph::polymorphed())
		buf += "\033[1;32mPoly ";
	coutRhsLine(2, buf);

	int ir = 3;
	for (map<unsigned char, Item>::iterator i = Inventory::items().begin(); i != Inventory::items().end() && ir < 50; ++i) {
		ostringstream ss;
		ss << "\033[m";
		if (i->second.beatitude() == BLESSED)
			ss << (unsigned char) 27 << "[32m";
		else if (i->second.beatitude() == CURSED)
			ss << (unsigned char) 27 << "[31m";
		else if (i->second.beatitude() == UNCURSED)
			ss << (unsigned char) 27 << "[33m";
		ss << i->first;
		ss << " - " << i->second;
		ss << (unsigned char) 27 << "[m";
		coutRhsLine(ir++, ss.str());
	}
	for (; ir < 50; ++ir)
		coutRhsLine(ir, "");
}

void World::forgetChanges() {
	for (vector<Point>::iterator c = _changes.begin(); c != _changes.end(); ++c)
		_changed[c->row()][c->col()] = false;
	_changes.clear();
}

bool World::executeCommand(const string& command) {
	/* send a command to nethack */
	_messages = "  "; // we want 2 spaces before the first message too
	if (command.size() <= 0) {
		/* huh? no command? */
		return false;
	}
	_connection->transmit(command);
	update();
	return true;
}

void World::fetchMenuText(int stoprow, int startcol, bool addspaces) {
	/* fetch text from a menu */
	for (int r = 0; r <= stoprow; ++r) {
		_msg_str = &_view[r][startcol];
		/* trim */
		string::size_type fns = _msg_str.find_first_not_of(" ");
		string::size_type lns = _msg_str.find_last_not_of(" ");
		if (fns == string::npos || lns == string::npos || fns >= lns)
			continue; // blank line?
		_msg_str = _msg_str.substr(fns, lns - fns + 1);
		if (addspaces)
			_msg_str.append(2, ' '); // append 2 spaces for later splitting
		_messages.append(_msg_str);
	}
}

void World::fetchMessages() {
	/* even yet a try on fetching messages sanely */
	_question = false; // we can do this as a question max last 1 turn
	// _msg_str = &_data[_data_size - sizeof (MORE)];
	// _msg_str = &_text[0];
	_msg_str = _view[_cursor.row()];
	Debug::info() << _msg_str << endl;
	string::size_type pos = string::npos;
	if ((pos = _msg_str.rfind(MORE, 0)) != string::npos) {
		/* "--More--" found */
		_menu = false; // we don't have a menu then
		int r = _cursor.row();
		int c = _cursor.col() - sizeof (MORE) + 1; // +1 because sizeof (MORE) is 9, not 8
		if (r == 0) {
			/* only one line, remove "--More--" from end of line */
			_msg_str = _view[r];
			_msg_str = _msg_str.substr(0, c);
			/* append 2 spaces for later splitting */
			_msg_str.append(2, ' ');
			_messages.append(_msg_str);
		} else {
			/* more than 1 line */
			if (c == 0 || _view[r][c - 1] != ' ') {
				/* this is just a very long line, not a list */
				c = 0;
				fetchMenuText(r, c, false);
				_messages.erase(_messages.size() - sizeof (MORE) + 1); // remove "--More--"
				_messages.append(2, ' '); // add two spaces
			} else {
				/* this is a list */
				/* sometimes "--More--" is placed 1 char to the right of the menu.
				 * this happens at least when the entire page is filled.
				 * check that the line above also is ' ', if not, c - 1 */
				if (_view[r - 1][c - 1] == ' ')
					fetchMenuText(r - 1, c, true); // "r - 1" to avoid the last "--More--"
				else
					fetchMenuText(r - 1, c - 1, true); // "r - 1" to avoid the last "--More--"
			}
		}
		/* request the remaining messages */
		_connection->transmit(" ");
		update();
		return;
	} else if (_cursor.row() == 0) {
		/* looks like we got a question.
		 * we might want to significantly improve this later,
		 * as we sometimes get partial data */
		_question = true;
		_menu = false; // no menu when we got a question
	} else {
		Debug::info() << _view[_cursor.row()] << endl;
		/* --More-- not found, but we might have a menu.
		 * this is pain */
		if (_menu) {
			/* we had a menu last frame, check if we still do */
			_msg_str = &_view[_last_menu.row()][_last_menu.col()];
			_cur_page = -1;
			_max_page = -1;
			if (
				_msg_str.find(END, 0) == string::npos
				&& sscanf(&_view[_last_menu.row()][_last_menu.col()], PAGE, &_cur_page, &_max_page) != 2
			) {
				/* nah, last menu is gone */
				_menu = false;
				_last_menu.row(-1);
				_last_menu.col(-1);
			} else {
				/* still got a menu */
				if (_cur_page == -1) {
					/* only 1 page */
					_cur_page = 1;
					_max_page = 1;
				}
			}
		}
		if (!_menu) {
			/* check if we got a new menu */
			// _msg_str = &_data[_data_size - sizeof (PAGE_DIRTY)];
			_msg_str = &_text[_cursor.row()*COLS];
			// _msg_str = _view[_cursor.row()];
			_cur_page = -1;
			_max_page = -1;
			if (_msg_str.find(END, 0) != string::npos || sscanf(_msg_str.c_str(), PAGE_DIRTY, &_cur_page, &_max_page) == 2) {
				/* hot jiggity! we got a list */
				/* now find the "(" in "(end) " or "(x of y)" */
				if (_cur_page == -1) {
					/* only 1 page */
					_cur_page = 1;
					_max_page = 1;
				}
				int c;
				for (c = _cursor.col(); c >= 0 && _view[_cursor.row()][c] != '('; --c)
					;
				_menu = true;
				_last_menu.row(_cursor.row());
				_last_menu.col(c);
			}
		}
		if (_menu) {
			/* finally parse the menu */
			fetchMenuText(_last_menu.row() - 1, _last_menu.col(), true); // "r - 1" to avoid the last "(end) " or "(x of y)"
		}
	}
	if (!_menu) {
		/* no "--More--", no question and no menu?
		 * well, it gotta be no messages or the message is on 1 line, then */
		_msg_str = _view[0];
		/* trim */
		string::size_type fns = _msg_str.find_first_not_of(" ");
		string::size_type lns = _msg_str.find_last_not_of(" ");
		if (fns == string::npos || lns == string::npos || fns >= lns)
			return; // blank line?
		_msg_str = _msg_str.substr(fns, lns - fns + 1);
		_messages.append(_msg_str);
		_messages.append(2, ' ');
	}
}

void World::handleEscapeSequence(int* pos, int* color) {
	if (_data[*pos] == 27) {
		/* sometimes we get 2 escape chars in a row,
		 * just return in those cases */
		return;
	} else if (_data[*pos] == '[') {
		int divider = -1;
		int start = *pos;
		for (; *pos < _data_size; ++*pos) {
			if (_data[*pos] == ';') {
				/* divider for values */
				divider = *pos;
			} else if (_data[*pos] == 'A') {
				/* move cursor up */
				if (_cursor.row() > 0)
					_cursor.moveNorth();
				break;
			} else if (_data[*pos] == 'B') {
				/* move cursor down */
				if (_cursor.row() < ROWS)
					_cursor.moveSouth();
				break;
			} else if (_data[*pos] == 'C') {
				/* move cursor right */
				if (_cursor.col() < COLS)
					_cursor.moveEast();
				break;
			} else if (_data[*pos] == 'D') {
				/* move cursor left */
				if (_cursor.col() > 0)
					_cursor.moveWest();
				break;
			} else if (_data[*pos] == 'H') {
				/* set cursor position */
				_cursor.row(0);
				_cursor.col(0);
				if (divider < 0)
					break;
				/* we got a position */
				int tmprow = _cursor.row();
				int tmpcol = _cursor.col();
				if (sscanf(&_data[start + 1], "%d;%d", &tmprow, &tmpcol) < 2) {
					Debug::error() << "Unable to place cursor: " << &_data[start] << endl;
					destroy();
					exit(13);
				}
				_cursor.row(--tmprow); // terminal starts counting from 1
				_cursor.col(--tmpcol); // ditto ^^
				break;
			} else if (_data[*pos] == 'J') {
				/* erase in display */
				if (_data[*pos - 1] == '[') {
					/* erase everything below current position */
					for (int r = _cursor.row() + 1; r < ROWS; ++r) {
						for (int c = 0; c < COLS; ++c) {
							_view[r][c] = ' ';
							addChangedLocation(Point(r,c));
						}
					}
				} else if (_data[*pos - 1] == '1') {
					/* erase everything above current position */
					for (int r = _cursor.row() - 1; r >= 0; --r) {
						for (int c = 0; c < COLS; ++c) {
							_view[r][c] = ' ';
							addChangedLocation(Point(r,c));
						}
					}
				} else if (_data[*pos - 1] == '2') {
					/* erase entire display */
					for (int r = 0; r < ROWS; ++r) {
						for (int c = 0; c < COLS; ++c) {
							_view[r][c] = ' ';
							addChangedLocation(Point(r,c));
						}
					}
					_cursor.row(0);
					_cursor.col(0);
					*color = 0;
				} else {
					Debug::error() << "Unhandled sequence: " << &_data[*pos] << endl;
					destroy();
					exit(9);
				}
				break;
			} else if (_data[*pos] == 'K') {
				/* erase in line */
				if (_data[*pos - 1] == '[') {
					/* erase everything to the right */
					for (int c = _cursor.col(); c < COLS; ++c) {
						_view[_cursor.row()][c] = ' ';
						addChangedLocation(Point(_cursor.row(),c));
					}
				} else if (_data[*pos - 1] == '1') {
					/* erase everything to the left */
					for (int c = 0; c < _cursor.col(); ++c) {
						_view[_cursor.row()][c] = ' ';
						addChangedLocation(Point(_cursor.row(),c));
					}
				} else if (_data[*pos - 1] == '2') {
					/* erase entire line */
					for (int c = 0; c < COLS; ++c) {
						_view[_cursor.row()][c] = ' ';
						addChangedLocation(Point(_cursor.row(),c));
					}
				} else {
					Debug::error() << "Unhandled sequence: " << &_data[*pos] << endl;
					destroy();
					exit(9);
				}
				break;
			} else if (_data[*pos] == 'h') {
				/* can possibly be ignored */
				/* probably [?1049h */
				break;
			} else if (_data[*pos] == 'l') {
				/* DEC Private Mode Reset? :s */
				break;
			} else if (_data[*pos] == 'm') {
				/* character attribute (bold, inverse, color, etc) */
				if (divider > 0) {
					Debug::error() << "Unsupported character color" << &_data[*pos] << endl;
					destroy();
					exit(15);
					break;
				}
				*color = 0;
				if (*pos == start + 1)
					break;
				int value = 0;
				int matched = sscanf(&_data[start + 1], "%d", &value);
				if (matched < 1) {
					Debug::error() << "Expected numeric value for character attribute: " << &_data[*pos] << endl;
					destroy();
					exit(14);
				}
				switch (value) {
				case NO_COLOR:
					_bold = false;
					_inverse = false;
					break;

				case BOLD:
					_bold = true;
					break;

				case INVERSE:
					_inverse = true;
					break;

				default:
					if (_bold)
						value += BOLD_OFFSET;
					if (_inverse)
						value += INVERSE_OFFSET;
					*color = value;
				}
				break;
			} else if (_data[*pos] == 'r') {
				/* this is some scrolling crap, ignore it */
				break;
			} else if (_data[*pos] == 'z') {
				/* this is generated by vt_tiledata; the stuff we care about is handled in Telnet */
				break;
			} else if (_data[*pos] == 27) {
				/* escape char found, that shouldn't happen */
				Debug::rawCharArray(_data, start, *pos + 1);
				destroy();
				exit(7);
			} else if (*pos - start > 7) {
				/* too long escape sequence? */
				Debug::error() << "Suspiciously long sequence: " << &_data[*pos] << endl;
				destroy();
				exit(8);
			}
		}
		if (*pos >= _data_size) {
			Debug::error()
			<< "Did not find stop char for sequence: "
			<< _data_size
			<< " "
			<< *pos
			<< " "
			<< _data
			<< endl;
			destroy();
			exit(6);
		}
	} else if (_data[*pos] == '(') {
		/* designate character set, ignore */
		++*pos;
	} else if (_data[*pos] == ')') {
		/* designate character set, ignore */
		++*pos;
	} else if (_data[*pos] == '*') {
		/* designate character set, ignore */
		++*pos;
	} else if (_data[*pos] == '+') {
		/* designate character set, ignore */
		++*pos;
	} else if (_data[*pos] == 'M') {
		/* reverse linefeed? */
		if (_cursor.row() > 0)
			_cursor.moveNorth();
	} else if (_data[*pos] == '=') {
		/* application numpad?
		 * ignore */
	} else if (_data[*pos] == '>') {
		/* normal numpad?
		 * ignore */
	} else {
		Debug::error() << "Unsupported escape sequence code at char " << *pos << ": " << &_data[*pos] << endl;
		destroy();
		exit(5);
	}
}

void World::update() {
	/* update the view */
	int charcolor = 0; // color of the char
	// while ((_data_size = _connection->retrieve(_data, BUFFER_SIZE)) <= 0)
	// 	sleep(1);
	// 	// usleep(20000);
	_data_size = _connection->retrieve(_data, BUFFER_SIZE);
	if (_data_size <= 0) {
		/* no data? sleep a sec and try again */
		sleep(1);
		_data_size = _connection->retrieve(_data, BUFFER_SIZE);
		if (_data_size <= 0) {
			Debug::error() << "No data received, quitting" << endl;
			destroy();
			exit(42);
		}
	}
	/* print world & data (to cerr, for debugging)
	 * this must be done here because if we get --More-- messages we'll update again */
	/* also, we do this in two loops because otherwise it flickers a lot */
	// Debug::rawCharArray(_data, 0, _data_size);
	memset(_text, 0, ROWS * COLS+1);

	_cout_last_color = -1;
	_cout_cursor.row(-1);
	for (int pos = 0; pos < _data_size; ++pos) {
		switch (_data[pos]) {
		case 0:
			/* sometimes we get lots of \0 characters.
			 * seemingly this happens when certain effects happen.
			 * for example, gas spore explotion, dagger thrown, etc.
			 * let's simply ignore these */
			break;
		case 8:
			/* backspace.
			 * make it go 1 char left */
			if (_cursor.col() > 0)
				_cursor.moveWest();
			break;

		case 10:
			/* line feed */
			_cursor.moveSouth();
			break;

		case 13:
			/* carriage return */
			_cursor.col(0);
			break;

		case 14:
			/* shift out, invoke G1 character set */
			break;

		case 15:
			/* shift in, invoke G0 character set */
			break;

		case 27:
			/* escape sequence coming up */
			++pos;
			handleEscapeSequence(&pos, &charcolor);
			break;

		default:
			/* add this char to the view */
			if (_cursor.col() >= COLS || _cursor.row() >= ROWS || _cursor.col() < 0 || _cursor.row() < 0) {
				Debug::warning() << "Fell out of the dungeon: " << _cursor.row() << ", " << _cursor.col() << endl;
				break;
			}
			_text[_cursor.row() * COLS + _cursor.col()] = \
				_view[_cursor.row()][_cursor.col()] = \
					(unsigned char) _data[pos];

			_color[_cursor.row()][_cursor.col()] = charcolor;
			addChangedLocation(_cursor);
			_cursor.moveEast();
			break;
		}
	}

	// Debug::info() << _cursor.row() << " " << _cursor.col() << endl;
	// Debug::info() << _view[0] << endl;
	// Debug::info() << _view[_cursor.row()] << endl;
	// Debug::info() << _text << endl;

	coutSetColor(NO_COLOR);
	coutGoto(_cursor.row()+1, _cursor.col()+1);
	cout.flush();

	fetchMessages();

	/* parse attribute & status rows */
	bool parsed_attributes = Saiph::parseAttributeRow(_view[ATTRIBUTES_ROW]);
	bool parsed_status = Saiph::parseStatusRow(_view[STATUS_ROW], _levelname, &_turn);
	/* check that the data we received seems ok */
	if (!_menu
		&& !_question
		&& (!parsed_attributes
			|| !parsed_status
			|| _cursor.row() < MAP_ROW_BEGIN
			|| _cursor.row() > MAP_ROW_END
			|| _cursor.col() < MAP_COL_BEGIN
			|| _cursor.col() > MAP_COL_END
	)) {
		/* hmm, what else can it be?
		 * could we be missing data?
		 * this is bad, we'll lose messages, this should never happen */
		Debug::warning() << "CURSOR ON UNEXPECTED LOCATION: " << _cursor.row() << ", " << _cursor.col() << endl;
		update();
		return;
	}
	++_frame_count;
	if (_messages == "  ")
		_messages.clear(); // no messages

	/* check if we get the question where we want to teleport */
	if (_messages.find(MESSAGE_FOR_INSTRUCTIONS, 0) != string::npos) {
		/* a bit unique case, this is a question.
		 * the data doesn't end with the sequence we check in World.
		 * however, the cursor is placed on the player when we get this message */
		_question = true;
	}

	/* check if we're engulfed */
	/* TODO: somehow this should be done in Saiph::analyze() */
	Saiph::engulfed(_cursor.insideMap() && _view[_cursor.row() - 1][_cursor.col() - 1] == '/' && _view[_cursor.row() - 1][_cursor.col() + 1] == '\\' && _view[_cursor.row() + 1][_cursor.col() - 1] == '\\' && _view[_cursor.row() + 1][_cursor.col() + 1] == '/');

	if (!_menu && !_question && !Saiph::engulfed())
		detectPosition();
}

/* main */
int main(int argc, const char* argv[]) {
	int connection_type = CONNECTION_TELNET;
	int initial_speed = SPEED_FAST;
	string logfile = "saiph.log";

	bool showUsage = false;
	bool t_o_r = false;
	if (argc > 1) {
		for (int a = 1; a < argc; ++a) {
			if (strlen(argv[a]) < 2) {
				showUsage = true;
				continue;
			}

			if (argv[a][0] == '-') {
				switch (argv[a][1]) {
				case 'h':
					showUsage = true;
					break;
				case 'l':
					connection_type = CONNECTION_LOCAL;
					break;
				case 't':
					connection_type = CONNECTION_TELNET;
					break;
				case 'E':
					connection_type = CONNECTION_EMBED;
					break;
				case '0':
					initial_speed = SPEED_PAUSE;
					break;
				case '2':
					initial_speed = SPEED_FAST;
					break;
				case '1':
					initial_speed = SPEED_SLOW;
					break;
				case 'R':
					t_o_r = true;
					connection_type = CONNECTION_REPLAY;
					if (argc > ++a)
						Replay::setFile(std::string(argv[a]), false);
					else
						showUsage = true;
					break;
				case 'T':
					t_o_r = true;
					if (argc > ++a)
						Replay::setFile(std::string(argv[a]), true);
					else
						showUsage = true;
					break;
				case 'L':
					if (argc > ++a)
						logfile = argv[a];
					else
						showUsage = true;
					break;
				default:
					cout << "Invalid argument " << argv[a] << endl;
					showUsage = true;
					break;
				}
			} else {
				cout << "Unknown argument specified." << endl;
			}
		}

		if (showUsage) {
			cout << "Usage: " << argv[0] << " [-l|-t] [-L <logfile>] [-R/-T <ttyrec>]" << endl;
			cout << endl;
			cout << "\t-l  Use local nethack executable" << endl;
			cout << "\t-t  Use telnet nethack server" << endl;
			cout << "\t-R  Replay saved nethack game" << endl;
			cout << "\t-E  Use in-process nethack interface" << endl;
			cout << "\t-T  Save raw game data during play" << endl;
			cout << endl;
			cout << "\t-0  Start paused" << endl;
			cout << "\t-1  Start at slow speed" << endl;
			cout << "\t-2  Start at full speed" << endl;
			cout << endl;
			cout << "\t-L <logfile>  Log file to write Saiph output" << endl;
			return 1;
		}
	}

	if (!t_o_r)
		Replay::setFile("saiph.ttyrec", true);

	/* init */
	World::init(logfile, connection_type);
	/* run */
	World::run(initial_speed);
	/* destroy */
	World::destroy();

}
