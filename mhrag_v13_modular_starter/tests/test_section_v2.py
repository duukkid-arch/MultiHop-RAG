"""Tests for strict heading-based, read-only section audit."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from mhrag_v13.open_document import SQLiteSectionReader
from mhrag_v13.controller import AdaptiveController
from mhrag_v13.contracts import Resolution, Study, Source, SearchResult
from mhrag_v13.verify import StructuralEvidenceGate
from mhrag_v13.answer import EvidencePacketAnswerer


class FakeResolver:
    def resolve(self, question):
        return Resolution((Study(44, 'BESS.pdf', 'BESS'), Study(55, 'FocusPro.pdf', 'Focus Pro')), (), ('BESS','Focus Pro'))


class FakeRetriever:
    def search(self, question, *, agentic=False, document_ids=None):
        return SearchResult([
            Source(44, 1001, '2', 'BESS.pdf', 'Methods. Randomized controlled clinical study of 232 infants and study outcomes.'),
            Source(55, 2001, '3', 'FocusPro.pdf', 'Methods. Randomized clinical trial studying adaptive digital therapy for ADHD.'),
        ], elapsed_ms=1)


class TestSectionAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'kb.sqlite'
        with sqlite3.connect(self.db) as conn:
            conn.executescript('''
                CREATE TABLE documents (id INTEGER, filename TEXT);
                CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, ordinal INTEGER, page INTEGER, body TEXT);
            ''')
            conn.executemany('INSERT INTO documents VALUES (?,?)', [(44,'BESS.pdf'),(55,'FocusPro.pdf'),(60,'Unknown.pdf')])
            entries = [
                (4402,44,20,2,'Methods\nStudy design and participants. The study enrolled children with bronchiolitis across sites. ' * 3),
                (4451,44,51,5,'Results\nFrom Dec 18 2018 to March 2024, infants with bronchiolitis were screened.'),
                (4452,44,52,5,'Results From Dec 18, 2018, to March 31, 2024, 1009 infants with bronchiolitis requiring ventilation were assessed; 232 were randomized to receive poractant alfa or sham intervention.' * 2),
                (4499,44,99,10,'Author contributions\nThis text discusses final Results as part of declarations only.' * 5),
                (5503,55,3,1,'Methods: In a single-centre, double-blind randomized trial, 85 children participated.' * 3),
                (5504,55,4,1,'Results: Both groups improved, and the combination group showed better attention outcome (p=0.006).' * 2),
                (5547,55,47,6,'Results\nBaseline characteristics and intention-to-treat results of the randomized trial were reported here.' * 4),
                (5577,55,77,12,'References\nReports of various intervention outcomes can be found in the citations.' * 3),
                (6001,60,1,1,'Author contributions and References. No primary clinical findings on this page.' * 3),
            ]
            conn.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',entries)
        self.reader = SQLiteSectionReader(self.db)

    def test_bess_results_found_body_not_contributors(self):
        rows = self.reader.audit_sections(44,sections=('results',))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['found'])
        self.assertEqual(rows[0]['chunk_id'],4452)
        self.assertEqual(rows[0]['page'],'5')
        self.assertTrue(rows[0]['local_open_only'])
        self.assertIn('232',rows[0]['excerpt'])

    def test_focus_results_prefers_body_to_abstract(self):
        rows=self.reader.audit_sections(55,sections=('results',))
        self.assertEqual(rows[0]['chunk_id'],5547)

    def test_results_missing_must_not_be_inferred(self):
        rows=self.reader.audit_sections(60,sections=('results',))
        self.assertEqual(rows[0]['status'],'heading_not_found')
        self.assertFalse(rows[0]['found'])

    def test_safe_invalid_inputs(self):
        with self.assertRaises(ValueError):self.reader.audit_sections(0)
        with self.assertRaises(ValueError):self.reader.audit_sections(44,sections=('references',))
        with self.assertRaises(ValueError):self.reader.audit_sections(44,per_section=0)

    def test_controller_keeps_local_audit_separate_from_api_evidence(self):
        controller=AdaptiveController(FakeResolver(),FakeRetriever(),StructuralEvidenceGate(),
              EvidencePacketAnswerer(), section_reader=self.reader)
        result=controller.run('比较 Focus Pro 和 BESS 的 Methods 与 Results')
        self.assertEqual(result['required_ids'],[44,55])
        self.assertEqual(result['observed_ids'],[44,55])
        self.assertEqual(len(result['section_audit']),4)
        self.assertTrue(all(e['local_open_only'] for e in result['section_audit']))
        self.assertTrue(all(e['found'] for e in result['section_audit']))
        self.assertFalse(result['claim_facts_verified'])
        self.assertTrue(all(s['origin']=='api' for s in result['sources']))
        self.assertEqual(result['http_calls'],1)

if __name__=='__main__':unittest.main()
