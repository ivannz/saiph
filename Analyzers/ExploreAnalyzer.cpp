#include "ExploreAnalyzer.h"

/* constructors */
ExploreAnalyzer::ExploreAnalyzer(Saiph *saiph) {
	this->saiph = saiph;
	place_count = 0;
	best_place = -1;
	symbols[symbol_count++] = FLOOR;
	symbols[symbol_count++] = CORRIDOR;
	symbols[symbol_count++] = OPEN_DOOR;
	symbols[symbol_count++] = STAIRS_UP;
	symbols[symbol_count++] = STAIRS_DOWN;
	symbols[symbol_count++] = ALTAR;
	symbols[symbol_count++] = GRAVE;
	symbols[symbol_count++] = THRONE;
	symbols[symbol_count++] = SINK;
	symbols[symbol_count++] = FOUNTAIN;
	symbols[symbol_count++] = ICE;
	symbols[symbol_count++] = LOWERED_DRAWBRIDGE;
	symbols[symbol_count++] = WEAPON;
	symbols[symbol_count++] = ARMOR;
	symbols[symbol_count++] = RING;
	symbols[symbol_count++] = AMULET;
	symbols[symbol_count++] = TOOL;
	symbols[symbol_count++] = FOOD;
	symbols[symbol_count++] = POTION;
	symbols[symbol_count++] = SCROLL;
	symbols[symbol_count++] = SPELLBOOK;
	symbols[symbol_count++] = WAND;
	symbols[symbol_count++] = GOLD;
	symbols[symbol_count++] = GEM;
	symbols[symbol_count++] = STATUE;
	symbols[symbol_count++] = IRON_BALL;
	symbols[symbol_count++] = PLAYER;
}

/* methods */
int ExploreAnalyzer::analyze(int row, int col, unsigned char symbol) {
	if (row <= MAP_ROW_START || row >= MAP_ROW_END || col <= 0 || col >= COLS - 1 || place_count >= EX_MAX_PLACES)
		return 0;
	int branch = saiph->current_branch;
	int dungeon = saiph->world->player.dungeon;
	/* are we in a corridor? */
	bool corridor = (saiph->branches[branch]->map[dungeon][row][col] == CORRIDOR);
	if ((!corridor && saiph->branches[branch]->search[dungeon][row][col] >= MAX_SEARCH) || (corridor && saiph->branches[branch]->search[dungeon][row][col] >= MAX_SEARCH * EX_DEAD_END_MULTIPLIER))
		return 0; // we've been here and searched frantically
	/* hjkl symbols */
	unsigned char hs = saiph->branches[branch]->map[dungeon][row][col - 1];
	unsigned char js = saiph->branches[branch]->map[dungeon][row + 1][col];
	unsigned char ks = saiph->branches[branch]->map[dungeon][row - 1][col];
	unsigned char ls = saiph->branches[branch]->map[dungeon][row][col + 1];
	/* hjkl unexplored */
	bool hune = (hs == SOLID_ROCK);
	bool june = (js == SOLID_ROCK);
	bool kune = (ks == SOLID_ROCK);
	bool lune = (ls == SOLID_ROCK);
	/* hjkl unpassable (only walls) */
	bool hunp = (hs == VERTICAL_WALL || hs == HORIZONTAL_WALL);
	bool junp = (js == VERTICAL_WALL || js == HORIZONTAL_WALL);
	bool kunp = (ks == VERTICAL_WALL || ks == HORIZONTAL_WALL);
	bool lunp = (ls == VERTICAL_WALL || ls == HORIZONTAL_WALL);
	/* *une || *unp */
	bool h = hune || hunp;
	bool j = june || junp;
	bool k = kune || kunp;
	bool l = lune || lunp;
	/* figure out score */
	int score = 0;
	bool dead_end = false;
	if (!corridor && (hune || june || kune || lune)) {
		score = EX_UNLIT_ROOM;
	} else if (corridor) {
		score = EX_CORRIDOR;
		if ((h && j && k) || (j && k && l) || (h && k && l) || (h && j && l))
			dead_end = true;
	} else if (!corridor && (hunp || junp || kunp || lunp)) {
		score = EX_EXTRA_SEARCH;
	}

	if (symbol == PLAYER && saiph->branches[branch]->map[dungeon][row][col] == CORRIDOR && !dead_end) {
		/* don't search in corridors unless it's a dead end */
		saiph->branches[branch]->search[dungeon][row][col] = MAX_SEARCH * EX_DEAD_END_MULTIPLIER;
		return 0;
	}
	if (score <= 0)
		return 0; // this place isn't very exciting
	places[place_count].row = row;
	places[place_count].col = col;
	places[place_count].move = ILLEGAL_MOVE;
	places[place_count].value = score;
	++place_count;
	return 0;
}

int ExploreAnalyzer::finish() {
	/* figure out which place to explore */
	int best = 0;
	int shortest_distance = -1;
	best_place = -1;
	for (int p = 0; p < place_count; ++p) {
		if (places[p].value < best && shortest_distance != -1)
			continue;
		int distance = 0;
		bool direct_line = false;
		places[p].move = saiph->shortestPath(places[p].row, places[p].col, false, distance, direct_line);
		if (places[p].move == ILLEGAL_MOVE)
			continue;
		if (places[p].value > best || (places[p].value == best && (shortest_distance == -1 || distance < shortest_distance))) {
			shortest_distance = distance;
			best = places[p].value;
			best_place = p;
		}
	}
	place_count = 0;

	if (best_place == -1)
		return best;

	if (places[best_place].move == REST)
		places[best_place].move = SEARCH; // search instead of rest
	return best;
}

void ExploreAnalyzer::command() {
	if (places[best_place].move == SEARCH)
		++saiph->branches[saiph->current_branch]->search[saiph->world->player.dungeon][places[best_place].row][places[best_place].col];
	char command[2];
	command[0] = places[best_place].move;
	command[1] = '\0';
	saiph->world->command(command);
}
