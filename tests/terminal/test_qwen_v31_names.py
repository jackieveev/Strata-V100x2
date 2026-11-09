"""#95: Unsloth's Dynamic v3.1 Qwen3.8-Flash-Next sets in the Terminal's model names: UD-IQ4_XS-v31 is its own name
(its Hub files are UD-IQ4_XS's, four shards), and the bare UD-IQ4_XS stays the pinned three-shard file."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, catalog

PC = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia"}
NEW = ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31")


class Names(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(PC))
        p.start()
        self.addCleanup(p.stop)

    def test_the_new_names_resolve_to_the_unsloth_family(self):
        cat = catalog(PC)
        for m in NEW:
            self.assertEqual(cat.resolve(m.lower()), ("unsloth", m))
            self.assertEqual(cat.resolve(m, "unsloth"), ("unsloth", m))
        self.assertEqual(cat.resolve("ud-iq4_xs"), ("unsloth", "UD-IQ4_XS"))

    def test_the_v31_iq4_xs_has_its_own_tag(self):
        cat = catalog(PC)
        self.assertEqual(cat.tag_of("unsloth", "UD-IQ4_XS-v31"), "unsloth-ud-iq4_xs-v31")
        self.assertEqual(cat.tag_of("unsloth", "UD-IQ4_XS"), "unsloth-ud-iq4_xs")


if __name__ == "__main__":
    unittest.main()
