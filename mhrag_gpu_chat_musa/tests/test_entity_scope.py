"""Regression tests for incorrect cross-trial evidence on real 4090 v11.2."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mhrag_gpu_chat_musa.core import ChatCore, named_study_terms


class WrongStudyFavoredEncoder:
    def embed(self, texts):
        result = np.zeros((len(texts), 1024), dtype=np.float32)
        result[:, 0] = 1.0
        return result

    def score(self, pairs):
        # Deliberately rank a *wrong* clinical study above BESS/GASH,
        # reproducing the real false H. pylori answer on v9_005.
        return [9.0 if "h. pylori" in text.lower() else 1.0
                for _, text in pairs]


class StudyScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.core = ChatCore(Path(self.temp.name) / "db.sqlite", WrongStudyFavoredEncoder())
        self.bess = self.core.ingest("doc44.txt", (
            "BESS is a multicentre phase 2 trial. The primary endpoint is "
            "duration of invasive mechanical ventilation until successful extubation. "
            "BESS treatment did not reduce ventilation time. " * 4).encode())["id"]
        self.gash = self.core.ingest("doc52.txt", (
            "GASH is a study protocol of glibenclamide for aneurysmal "
            "subarachnoid haemorrhage. GASH primary outcome is mRS at six months. "
            "The study protocol does not present final trial efficacy results. " * 4).encode())["id"]
        self.other = self.core.ingest("doc13.txt", (
            "An H. pylori trial. The primary endpoint of this study was the "
            "eradication rate of H. pylori. Successful eradication used a breath test. "
            * 4).encode())["id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_named_entity_extraction(self):
        self.assertEqual(named_study_terms("比较GASH与BESS的主要终点"), ["GASH", "BESS"])
        self.assertEqual(named_study_terms("两篇NeuroAI文章"), ["NeuroAI"])
        self.assertEqual(named_study_terms("随机试验主要终点是什么"), [])

    def test_agent_does_not_confuse_bess_with_h_pylori(self):
        question = "在BESS试验中主要终点是什么？"
        baseline, _ = self.core.retrieve("BESS trial main endpoint", "hybrid_ce")
        self.assertEqual(baseline[0]["document_id"], self.other)
        found, trace = self.core.agent_tool_search(
            "BESS trial main endpoint", "hybrid_ce", original_question=question)
        self.assertTrue(found)
        self.assertEqual({s["document_id"] for s in found}, {self.bess})
        self.assertEqual(trace["entity_scope"], [self.bess])
        self.assertEqual(trace["entity_names"], ["BESS"])
        self.assertTrue(trace["original_query_fused"])

    def test_multi_study_scope_is_union_not_intersection(self):
        sources, trace = self.core.agent_tool_search(
            "GASH endpoint BESS endpoint", "hybrid_ce",
            original_question="比较GASH与BESS试验的主要终点", limit=6)
        self.assertEqual(trace["entity_scope"], sorted([self.bess, self.gash]))
        self.assertTrue({s["document_id"] for s in sources} <= {self.bess, self.gash})

    def test_explicit_document_restriction_is_never_bypassed(self):
        sources, trace = self.core.agent_tool_search(
            "BESS endpoint", "hybrid_ce", doc_ids=[self.other],
            original_question="BESS研究的主要终点")
        self.assertEqual(sources, [])
        self.assertEqual(trace["entity_scope"], [])

    def test_unknown_entity_falls_back_to_normal_search(self):
        _, trace = self.core.agent_tool_search(
            "XZZZ trial main endpoint", "hybrid_ce",
            original_question="XZZZ试验主要终点")
        self.assertIsNone(trace["entity_scope"])
        self.assertEqual(trace["entity_names"], [])


if __name__ == "__main__":
    unittest.main()
