#include "Saiph.h"

#include <cstdio>
#include <string.h>
#include "Globals.h"
#include "Inventory.h"
#include "World.h"
#include "Data/Skill.h"
#include "Debug.h"

#define MESSAGE_SPEED_GAIN1 "  You feel quick!  "
#define MESSAGE_SPEED_GAIN2 "  You seem faster.  "
#define MESSAGE_SPEED_GAIN3 "  You speed up.  "
#define MESSAGE_SPEED_GAIN4 "  Your quickness feels more natural.  "
#define MESSAGE_SPEED_GAIN5 "  \"and thus I grant thee the gift of Speed!\"  "

#define MESSAGE_SPEED_LOSE1 "  You feel slow!  "
#define MESSAGE_SPEED_LOSE2 "  You seem slower.  "
#define MESSAGE_SPEED_LOSE3 "  You feel slower.  "
#define MESSAGE_SPEED_LOSE4 "  You are slowing down.  "
#define MESSAGE_SPEED_LOSE5 "  Your limbs are getting oozy.  "
#define MESSAGE_SPEED_LOSE6 "  You slow down.  "
#define MESSAGE_SPEED_LOSE7 "  Your quickness feels less natural.  "

#define MESSAGE_VERYFAST_LOSE1 "  You slow down.  "
#define MESSAGE_VERYFAST_LOSE2 "  Your quickness feels less natural.  "
#define MESSAGE_VERYFAST_LOSE3 "  You feel yourself slowing down.  "
#define MESSAGE_VERYFAST_LOSE4 "  You feel yourself slowing down a bit.  "

#define MESSAGE_VERYFAST_GAIN1 "  You are suddenly moving faster.  "
#define MESSAGE_VERYFAST_GAIN2 "  You are suddenly moving much faster.  "
#define MESSAGE_VERYFAST_GAIN3 "  Your knees seem more flexible now.  "


using namespace analyzer;
using namespace std;

/* variables */
/* attributes */
int Saiph::_alignment = NEUTRAL; // see defined constants
int Saiph::_charisma = 0;
int Saiph::_constitution = 0;
int Saiph::_dexterity = 0;
int Saiph::_intelligence = 0;
int Saiph::_strength = 0;
int Saiph::_wisdom = 0;
/* status */
int Saiph::_armor = 0;
int Saiph::_encumbrance = UNENCUMBERED; // see defined constants
int Saiph::_experience = 0;
int Saiph::_hunger = CONTENT; // see defined constants
int Saiph::_hitpoints = 0;
int Saiph::_hitpoints_max = 0;
int Saiph::_power = 0;
int Saiph::_power_max = 0;
int Saiph::_min_moves_this_turn = 1;
/* effects */
bool Saiph::_blind = false;
bool Saiph::_confused = false;
bool Saiph::_foodpoisoned = false;
bool Saiph::_hallucinating = false;
bool Saiph::_ill = false;
bool Saiph::_slimed = false;
bool Saiph::_stoned = false;
bool Saiph::_stunned = false;
bool Saiph::_hurt_leg = false;
bool Saiph::_polymorphed = false;
bool Saiph::_engulfed = false;
bool Saiph::_in_a_pit = false;
/* position */
Coordinate Saiph::_position;
/* zorkmids */
int Saiph::_zorkmids = 0;
/* conducts */
unsigned long long Saiph::_conducts = 0;
/* intrinsics/extrinsics */
unsigned long long Saiph::_intrinsics = 0;
unsigned long long Saiph::_extrinsics = 0;
/* current skills */
int Saiph::_current_skills[P_NUM_SKILLS];
/* last turn she prayed */
int Saiph::_last_prayed = 0;
/* name */
string Saiph::_name;
/* race */
string Saiph::_race;
/* role */
int Saiph::_role = UNKNOWN_ROLE;
/* gender */
int Saiph::_gender = UNKNOWN_GENDER;
/* effects */
char Saiph::_effects[MAX_EFFECTS][MAX_TEXT_LENGTH] = {
	{'\0'}
};

/* methods */
void Saiph::analyze() {
}

void Saiph::parseMessages(const string& messages) {
	if (World::menu()) {
		if (messages.find(MESSAGE_BASE_ATTRIBUTES) != string::npos) {
			/* name */
			string::size_type pos = messages.find(':');
			if (pos != string::npos)
				_name = messages.substr(pos + 2, messages.find("  ", pos + 2) - pos - 2);
			/* skip starting race/role/gender/alignment */
			pos = messages.find(':', pos + 2);
			pos = messages.find(':', pos + 2);
			pos = messages.find(':', pos + 2);
			pos = messages.find(':', pos + 2);
			/* current race */
			pos = messages.find(':', pos + 2);
			if (pos != string::npos)
				_race = messages.substr(pos + 2, messages.find("  ", pos + 2) - pos - 2);
			/* current role */
			pos = messages.find(':', pos + 2);
			if (pos != string::npos) {
				string role = messages.substr(pos + 2, messages.find("  ", pos + 2) - pos - 2);
				if (role == "Archeologist") {
					_role = ARCHEOLOGIST;
					_intrinsics |= PROPERTY_SPEED;
					_intrinsics |= PROPERTY_STEALTH;
				} else if (role == "Barbarian") {
					_role = BARBARIAN;
					_intrinsics |= PROPERTY_POISON;
				} else if (role == "Caveman") {
					_role = CAVEMAN;
				} else if (role == "Healer") {
					_role = HEALER;
					_intrinsics |= PROPERTY_POISON;
				} else if (role == "Knight") {
					_role = KNIGHT;
				} else if (role == "Monk") {
					_role = MONK;
					_intrinsics |= PROPERTY_SLEEP;
					_intrinsics |= PROPERTY_SPEED;
					_conducts |= CONDUCT_VEGETARIAN;
				} else if (role == "Priest") {
					_role = PRIEST;
				} else if (role == "Ranger") {
					_role = RANGER;
				} else if (role == "Rogue") {
					_role = ROGUE;
					_intrinsics |= PROPERTY_STEALTH;
				} else if (role == "Samurai") {
					_role = SAMURAI;
					_intrinsics |= PROPERTY_SPEED;
				} else if (role == "Tourist") {
					_role = TOURIST;
				} else if (role == "Valkyrie") {
					_role = VALKYRIE;
					_intrinsics |= PROPERTY_COLD;
					_intrinsics |= PROPERTY_STEALTH;
				} else if (role == "Wizard") {
					_role = WIZARD;
				} else {
					_role = UNKNOWN_ROLE;
				}
			}
			/* current gender */
			pos = messages.find(':', pos + 2);
			if (pos != string::npos) {
				string gender = messages.substr(pos + 2, messages.find("  ", pos + 2) - pos - 2);
				if (gender == "female")
					_gender = FEMALE;
				else if (gender == "male")
					_gender = MALE;
				else if (gender == "neuter")
					_gender = NEUTER;
				else
					_gender = UNKNOWN_GENDER;
			}
		}
	} else {
		if (messages.find(MESSAGE_COLD_RES_GAIN1) != string::npos)
			_intrinsics |= PROPERTY_COLD;
		if (messages.find(MESSAGE_COLD_RES_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_COLD;
		if (messages.find(MESSAGE_DISINTEGRATION_RES_GAIN1) != string::npos || messages.find(MESSAGE_DISINTEGRATION_RES_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_DISINT;
		if (messages.find(MESSAGE_FIRE_RES_GAIN1) != string::npos || messages.find(MESSAGE_FIRE_RES_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_FIRE;
		if (messages.find(MESSAGE_FIRE_RES_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_FIRE;
		if (messages.find(MESSAGE_POISON_RES_GAIN1) != string::npos || messages.find(MESSAGE_POISON_RES_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_POISON;
		if (messages.find(MESSAGE_POISON_RES_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_POISON;
		if (messages.find(MESSAGE_SHOCK_RES_GAIN1) != string::npos || messages.find(MESSAGE_SHOCK_RES_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_SHOCK;
		if (messages.find(MESSAGE_SHOCK_RES_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_SHOCK;
		if (messages.find(MESSAGE_SLEEP_RES_GAIN1) != string::npos)
			_intrinsics |= PROPERTY_SLEEP;
		if (messages.find(MESSAGE_SLEEP_RES_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_SLEEP;
		if (messages.find(MESSAGE_TELEPATHY_GAIN1) != string::npos)
			_intrinsics |= PROPERTY_ESP;
		if (messages.find(MESSAGE_TELEPATHY_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_ESP;
		if (messages.find(MESSAGE_TELEPORT_CONTROL_GAIN1) != string::npos || messages.find(MESSAGE_TELEPORT_CONTROL_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_TELEPORT_CONTROL;
		if (messages.find(MESSAGE_TELEPORTITIS_GAIN1) != string::npos || messages.find(MESSAGE_TELEPORTITIS_GAIN2) != string::npos)
			_intrinsics |= PROPERTY_TELEPORT;
		if (messages.find(MESSAGE_TELEPORTITIS_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_TELEPORT;
		if (messages.find(MESSAGE_LYCANTHROPY_GAIN1) != string::npos)
			_intrinsics |= PROPERTY_LYCANTHROPY;
		if (messages.find(MESSAGE_LYCANTHROPY_LOSE1) != string::npos)
			_intrinsics &= ~PROPERTY_LYCANTHROPY;
		if (messages.find(MESSAGE_SPEED_GAIN1) != string::npos || messages.find(MESSAGE_SPEED_GAIN2) != string::npos || messages.find(MESSAGE_SPEED_GAIN3) != string::npos || messages.find(MESSAGE_SPEED_GAIN4) != string::npos || messages.find(MESSAGE_SPEED_GAIN5) != string::npos)
			_intrinsics |= PROPERTY_SPEED;
		if (messages.find(MESSAGE_SPEED_LOSE1) != string::npos || messages.find(MESSAGE_SPEED_LOSE2) != string::npos || messages.find(MESSAGE_SPEED_LOSE3) != string::npos || messages.find(MESSAGE_SPEED_LOSE4) != string::npos || messages.find(MESSAGE_SPEED_LOSE5) != string::npos || messages.find(MESSAGE_SPEED_LOSE6) != string::npos || messages.find(MESSAGE_SPEED_LOSE7) != string::npos)
			_intrinsics &= ~PROPERTY_SPEED;
		if (messages.find(MESSAGE_VERYFAST_GAIN1) != string::npos || messages.find(MESSAGE_VERYFAST_GAIN2) != string::npos || messages.find(MESSAGE_VERYFAST_GAIN3) != string::npos)
			_extrinsics |= PROPERTY_VERYFAST;
		if (messages.find(MESSAGE_VERYFAST_LOSE1) != string::npos || messages.find(MESSAGE_VERYFAST_LOSE2) != string::npos || messages.find(MESSAGE_VERYFAST_LOSE3) != string::npos || messages.find(MESSAGE_VERYFAST_LOSE4) != string::npos)
			_extrinsics &= ~PROPERTY_VERYFAST;
		if (messages.find(MESSAGE_SLOWING_DOWN) != string::npos || messages.find(MESSAGE_LIMBS_ARE_STIFFENING) != string::npos)
			_stoned = true; // not checking for limbs turned to stone because we're dead then
		if (messages.find(MESSAGE_YOU_FEEL_LIMBER) != string::npos)
			_stoned = false;
		if (messages.find(MESSAGE_HURT_LEFT_LEG) != string::npos || messages.find(MESSAGE_HURT_RIGHT_LEG) != string::npos)
			_hurt_leg = true;
		if (messages.find(MESSAGE_LEG_IS_BETTER) != string::npos)
			_hurt_leg = false;
		if (messages.find(MESSAGE_POLYMORPH) != string::npos)
			_polymorphed = true;
		if (messages.find(MESSAGE_UNPOLYMORPH) != string::npos)
			_polymorphed = false;
		if (messages.find(MESSAGE_LEVITATION_GAIN1) != string::npos || messages.find(MESSAGE_LEVITATION_GAIN2) != string::npos)
			_extrinsics |= PROPERTY_LEVITATION;
		if (messages.find(MESSAGE_LEVITATION_LOSE1) != string::npos || messages.find(MESSAGE_LEVITATION_LOSE2) != string::npos)
			_extrinsics &= ~PROPERTY_LEVITATION;
		if (messages.find(MESSAGE_CANT_REACH_OVER_PIT) != string::npos || messages.find(MESSAGE_STILL_IN_PIT) != string::npos || messages.find(MESSAGE_FALL_INTO_PIT) != string::npos || messages.find(MESSAGE_YOU_DIG_A_PIT) != string::npos)
			_in_a_pit = true;
		if (messages.find(MESSAGE_CRAWL_OUT_OF_PIT) != string::npos || messages.find(MESSAGE_YOU_FLOAT_OUT_OF_PIT) != string::npos || messages.find(MESSAGE_CANNOT_REACH_BOTTOM_OF_PIT) != string::npos)
			_in_a_pit = false;
	}
}

bool Saiph::parseAttributeRow(const char* attributerow) {
	/* fetch attributes */
	int matched = sscanf(attributerow, "%*[^:]:%d%*[^:]:%d%*[^:]:%d%*[^:]:%d%*[^:]:%d%*[^:]:%d%s", &_strength, &_dexterity, &_constitution, &_intelligence, &_wisdom, &_charisma, _effects[0]);
	if (matched < 7)
		return false;
	if (_effects[0][0] == 'L')
		_alignment = LAWFUL;
	else if (_effects[0][0] == 'N')
		_alignment = NEUTRAL;
	else
		_alignment = CHAOTIC;
	return true;
}

bool Saiph::parseStatusRow(const char* statusrow, char* levelname, int* turn) {
	/* fetch status */
	// int matched = sscanf(statusrow, "%16[^$*]%*[^:]:%d%*[^:]:%d(%d%*[^:]:%d(%d%*[^:]:%d%*[^:]:%d%*[^:]:%d%s%s%s%s%s", levelname, &_zorkmids, &_hitpoints, &_hitpoints_max, &_power, &_power_max, &_armor, &_experience, turn, _effects[0], _effects[1], _effects[2], _effects[3], _effects[4]);
	// if (matched < 9)
	int matched = sscanf(statusrow, "%16[^$*]%*[^:]:%d%*[^:]:%d(%d%*[^:]:%d(%d%*[^:]:%d%*[^:]:%d%*[^:]%s%s%s%s%s", levelname, &_zorkmids, &_hitpoints, &_hitpoints_max, &_power, &_power_max, &_armor, &_experience, _effects[0], _effects[1], _effects[2], _effects[3], _effects[4]);

	// XXX potentially problematic
	*turn = 1;

	if (matched < 8)
		return false;
	_encumbrance = UNENCUMBERED;
	_hunger = CONTENT;
	_blind = false;
	_confused = false;
	_foodpoisoned = false;
	_hallucinating = false;
	_ill = false;
	_slimed = false;
	_stunned = false;
	int effects_found = matched - 9;
	for (int e = 0; e < effects_found; ++e) {
		if (strcmp(_effects[e], "Burdened") == 0) {
			_encumbrance = BURDENED;
		} else if (strcmp(_effects[e], "Stressed") == 0) {
			_encumbrance = STRESSED;
		} else if (strcmp(_effects[e], "Strained") == 0) {
			_encumbrance = STRAINED;
		} else if (strcmp(_effects[e], "Overtaxed") == 0) {
			_encumbrance = OVERTAXED;
		} else if (strcmp(_effects[e], "Overloaded") == 0) {
			_encumbrance = OVERLOADED;
		} else if (strcmp(_effects[e], "Fainting") == 0) {
			_hunger = FAINTING;
		} else if (strcmp(_effects[e], "Fainted") == 0) {
			_hunger = FAINTING;
		} else if (strcmp(_effects[e], "Weak") == 0) {
			_hunger = WEAK;
		} else if (strcmp(_effects[e], "Hungry") == 0) {
			_hunger = HUNGRY;
		} else if (strcmp(_effects[e], "Satiated") == 0) {
			_hunger = SATIATED;
		} else if (strcmp(_effects[e], "Oversatiated") == 0) {
			_hunger = OVERSATIATED;
		} else if (strcmp(_effects[e], "Blind") == 0) {
			_blind = true;
		} else if (strcmp(_effects[e], "Conf") == 0) {
			_confused = true;
		} else if (strcmp(_effects[e], "FoodPois") == 0) {
			_foodpoisoned = true;
		} else if (strcmp(_effects[e], "Hallu") == 0) {
			_hallucinating = true;
		} else if (strcmp(_effects[e], "Ill") == 0) {
			_ill = true;
		} else if (strcmp(_effects[e], "Slime") == 0) {
			_slimed = true;
		} else if (strcmp(_effects[e], "Stun") == 0) {
			_stunned = true;
		}
	}
	return true;
}

int Saiph::alignment() {
	return _alignment;
}

int Saiph::charisma() {
	return _charisma;
}

int Saiph::constitution() {
	return _constitution;
}

int Saiph::dexterity() {
	return _dexterity;
}

int Saiph::intelligence() {
	return _intelligence;
}

int Saiph::strength() {
	return _strength;
}

int Saiph::wisdom() {
	return _wisdom;
}

int Saiph::armor() {
	return _armor;
}

int Saiph::encumbrance() {
	return _encumbrance;
}

int Saiph::experience() {
	return _experience;
}

int Saiph::hunger() {
	return _hunger;
}

int Saiph::hitpoints() {
	return _hitpoints;
}

int Saiph::hitpointsMax() {
	return _hitpoints_max;
}

int Saiph::power() {
	return _power;
}

int Saiph::powerMax() {
	return _power_max;
}

bool Saiph::blind() {
	return _blind;
}

bool Saiph::confused() {
	return _confused;
}

bool Saiph::foodpoisoned() {
	return _foodpoisoned;
}

bool Saiph::hallucinating() {
	return _hallucinating;
}

bool Saiph::ill() {
	return _ill;
}

bool Saiph::slimed() {
	return _slimed;
}

bool Saiph::stoned() {
	return _stoned;
}

bool Saiph::stunned() {
	return _stunned;
}

bool Saiph::hurtLeg() {
	return _hurt_leg;
}

bool Saiph::hurtLeg(bool hurt_leg) {
	_hurt_leg = hurt_leg;
	return Saiph::hurtLeg();
}

bool Saiph::polymorphed() {
	return _polymorphed;
}

bool Saiph::polymorphed(bool polymorphed) {
	_polymorphed = polymorphed;
	return Saiph::polymorphed();
}

bool Saiph::infravision() {
	// TODO some polymorph forms have it
	return race() != "human" && !polymorphed();
}

bool Saiph::engulfed() {
	return _engulfed;
}

bool Saiph::engulfed(bool engulfed) {
	_engulfed = engulfed;
	return Saiph::engulfed();
}

bool Saiph::inAPit() {
	return _in_a_pit;
}

int Saiph::skill(int which) {
	return _current_skills[which] ? _current_skills[which] : P_UNSKILLED;
}

int Saiph::maxSkill(int which) {
	int rmax = data::Skill::roleMax(role(), which);
	/* account for divine unrestriction */
	return (rmax > P_UNSKILLED) ? rmax : _current_skills[which] ? P_BASIC : P_UNSKILLED;
}

void Saiph::updateSkills(int* curp) {
	for (int i = 0; i < P_NUM_SKILLS; ++i)
		_current_skills[i] = curp[i];
}

const Coordinate& Saiph::position() {
	return _position;
}

const Coordinate& Saiph::position(const Coordinate& position) {
	if (position != _position)
		_in_a_pit = false; // sometimes she gets out of a pit without any message about it (teleportitis?)
	_position = position;
	return Saiph::position();
}

int Saiph::zorkmids() {
	return _zorkmids;
}

unsigned long long Saiph::conducts() {
	return _conducts;
}

unsigned long long Saiph::addConducts(unsigned long long conducts) {
	_conducts |= conducts;
	return Saiph::conducts();
}

unsigned long long Saiph::removeConducts(unsigned long long conducts) {
	_conducts &= ~conducts;
	return Saiph::conducts();
}

unsigned long long Saiph::intrinsics() {
	return _intrinsics;
}

unsigned long long Saiph::addIntrinsics(unsigned long long intrinsics) {
	_intrinsics |= intrinsics;
	return Saiph::intrinsics();
}

unsigned long long Saiph::removeIntrinsics(unsigned long long intrinsics) {
	_intrinsics &= ~intrinsics;
	return Saiph::intrinsics();
}

unsigned long long Saiph::extrinsics() {
	return _extrinsics | Inventory::extrinsicsFromItems();
}

unsigned long long Saiph::addExtrinsics(unsigned long long extrinsics) {
	_extrinsics |= extrinsics;
	return Saiph::extrinsics();
}

unsigned long long Saiph::removeExtrinsics(unsigned long long extrinsics) {
	_extrinsics &= ~extrinsics;
	return Saiph::extrinsics();
}

int Saiph::lastPrayed() {
	return _last_prayed;
}

int Saiph::lastPrayed(int last_prayed) {
	_last_prayed = last_prayed;
	return Saiph::lastPrayed();
}

const string& Saiph::name() {
	return _name;
}

const string& Saiph::race() {
	return _race;
}

int Saiph::role() {
	return _role;
}

int Saiph::gender() {
	return _gender;
}

int Saiph::minMovesThisTurn() {
	return _min_moves_this_turn;
}

int Saiph::minMovesThisTurn(int set_to) {
	_min_moves_this_turn = set_to;
	return _min_moves_this_turn;
}

// NOT HANDLED: polyself, riding
int Saiph::minSpeed() {
	int moveamt = (Saiph::extrinsics() & PROPERTY_VERYFAST) ? 18 : 12;
	switch (_encumbrance) {
		case BURDENED:
			moveamt -= (moveamt / 4);
			break;

		case STRESSED:
			moveamt -= (moveamt / 2);
			break;

		case STRAINED:
			moveamt -= ((moveamt * 3) / 4);
			break;

		case OVERTAXED:
			moveamt -= ((moveamt * 7) / 8);
			break;

		default:
			break;
	}
	return moveamt;
}

int Saiph::maxSpeed() {
	int moveamt = (Saiph::extrinsics() & PROPERTY_VERYFAST) ? 24 : (Saiph::intrinsics() & PROPERTY_SPEED) ? 18 : 12;
	switch (_encumbrance) {
		case BURDENED:
			moveamt -= (moveamt / 4);
			break;

		case STRESSED:
			moveamt -= (moveamt / 2);
			break;

		case STRAINED:
			moveamt -= ((moveamt * 3) / 4);
			break;

		case OVERTAXED:
			moveamt -= ((moveamt * 7) / 8);
			break;

		default:
			break;
	}
	return moveamt;
}
