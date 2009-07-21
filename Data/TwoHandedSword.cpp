#include "TwoHandedSword.h"

using namespace data;
using namespace std;

/* initialize static variables */
map<string, TwoHandedSword *> TwoHandedSword::two_handed_swords;

TwoHandedSword::TwoHandedSword(const string &name, int cost, int weight, char item_class, int material, unsigned long long properties, bool one_handed, int alignment, const Attack &as0, const Attack &as1, const Attack &as2, const Attack &al0, const Attack &al1, const Attack &al2) : Weapon(name, cost, weight, item_class, material, properties, one_handed, alignment, as0, as1, as2, al0, al1, al2) {
}

void TwoHandedSword::addToMap(const string &name, TwoHandedSword *two_handed_sword) {
	TwoHandedSword::two_handed_swords[name] = two_handed_sword;
	Weapon::addToMap(name, two_handed_sword);
}

void TwoHandedSword::create(const string &name, int cost, int weight, const Attack &as0, const Attack &as1, const Attack &as2, const Attack &al0, const Attack &al1, const Attack &al2, int material, char item_class, unsigned long long properties, bool one_handed, int alignment) {
	addToMap(name, new TwoHandedSword(name, cost, weight, item_class, material, properties, one_handed, alignment, as0, as1, as2, al0, al1, al2));
}

void TwoHandedSword::init() {
	/* two-handed swords */
	create("two-handed sword", 50, 150, Attack(AT_CLAW, AD_PHYS, 1, 12), Attack(), Attack(), Attack(AT_CLAW, AD_PHYS, 3, 6), Attack(), Attack(), MATERIAL_IRON, ')', 0, false, CHAOTIC | NEUTRAL | LAWFUL);
	create("tsurugi", 500, 60, Attack(AT_CLAW, AD_PHYS, 1, 16), Attack(), Attack(), Attack(AT_CLAW, AD_PHYS, 1, 8), Attack(AT_CLAW, AD_PHYS, 2, 6), Attack(), MATERIAL_METAL, ')', 0, false, CHAOTIC | NEUTRAL | LAWFUL);

	/* aliases */
	addToMap("long samurai sword", two_handed_swords["tsurugi"]);

	/* artifact two-handed swords */
	create("The Tsurugi of Muramasa", 4500, 60, Attack(AT_CLAW, AD_PHYS, 1, 16), Attack(AT_CLAW, AD_PHYS, 1, 8), Attack(), Attack(AT_CLAW, AD_PHYS, 1, 8), Attack(AT_CLAW, AD_PHYS, 2, 6), Attack(AT_CLAW, AD_PHYS, 1, 8), MATERIAL_METAL, ')', PROPERTY_MAGIC | PROPERTY_ARTIFACT, false, LAWFUL);
}