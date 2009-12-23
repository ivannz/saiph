#ifndef ANALYZER_VALKYRIE_H
#define ANALYZER_VALKYRIE_H

#include "Request.h"
#include "Analyzers/Analyzer.h"

class Saiph;

namespace analyzer {
	class Valkyrie : public Analyzer {
	public:
		Valkyrie(Saiph* saiph);

		void init();

	private:
		Saiph* saiph;
		Request req;
		unsigned char loot_group;

		void setupAmulet();
		void setupArmor();
		void setupFood();
		void setupPotion();
		void setupRing();
		void setupTool();
		void setupWand();
		void setupWeapon();
	};
}
#endif
