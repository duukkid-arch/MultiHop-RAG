"""Regression tests derived from observed AutoDL BESS / Focus Pro split patterns."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from mhrag_v13.open_document import SQLiteSectionReader


class SectionReaderV3Test(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db = Path(temp.name) / 'fixture.sqlite'
        with sqlite3.connect(db) as con:
            con.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, ordinal INTEGER, page INTEGER, body TEXT)')
            con.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)', [
                (4407,44,7,1,'Summary of randomized trial.\nFindings The trial was completed after six recruitment seasons. 1009 infants were assessed for eligibility and 232 randomized. ' * 2),
                (4451,44,51,5,'A'*621+'\nResults From Dec 18 2018 to March 2024, 1009 infants were assessed for eligibility. ' * 4),
                (4452,44,52,5,'A'*31+'\nResults From Dec 18, 2018 to March 31, 2024, 1009 infants were assessed and 232 randomized to receive surfactant or sham. ' * 4),
                (4487,44,87,9,'Discussion of the study.\nfindings. The sham control intervention addressed placebo effects. ' * 5),
                (5504,55,4,1,'Abstract\nBackground text.\nResults: Both groups showed comparable improvements in subjective measures, but combination had greater gains (p=0.006; d=0.809). ' * 3),
                (5547,55,47,6,'B'*259+'\nResults Baseline Characteristics. Intent-to-treat analysis included the 85 children randomized to two groups. ' * 4),
                (5591,55,91,12,'Discussion\nfindings may not be generalized to broader populations. A larger multicenter trial is needed. ' * 5),
                (5592,55,92,13,'Author contributions\nResults The contribution of each researcher is discussed here. ' * 5),
                (6600,66,1,1,'Literature review of recent findings and methods, without actual findings of any trials. ' * 3),
            ])
        self.reader = SQLiteSectionReader(db)

    def test_bess_results_inside_later_lines(self):
        r = self.reader.audit_sections(44,sections=('results',))[0]
        self.assertTrue(r['found'])
        self.assertEqual(r['chunk_id'],4452)
        self.assertIn('1009',r['excerpt'])
        self.assertTrue(r['heading_offset'] > 0)
        self.assertEqual(r['section_candidate_type'],'body_candidate')

    def test_focus_prefers_real_body_over_abstract_and_discussion(self):
        r = self.reader.audit_sections(55, sections=('results',))[0]
        self.assertTrue(r['found'])
        self.assertEqual(r['chunk_id'],5547)
        self.assertIn('Baseline Characteristics',r['excerpt'])
        self.assertNotIn('findings may not',r['excerpt'])

    def test_abstract_is_available_as_second_option(self):
        rows=self.reader.audit_sections(55,sections=('results',),per_section=2)
        self.assertEqual([r['chunk_id'] for r in rows],[5547,5504])

    def test_no_fabricated_results(self):
        r=self.reader.audit_sections(66,sections=('results',))[0]
        self.assertFalse(r['found'])
        self.assertEqual(r['status'],'heading_not_found')
        self.assertTrue(r['local_open_only'])

    def test_no_references_discussion_leak(self):
        rows=self.reader.audit_sections(44,sections=('results',),per_section=3)
        self.assertTrue(all(x['chunk_id'] != 4487 for x in rows))
        rows=self.reader.audit_sections(55,sections=('results',),per_section=3)
        self.assertTrue(all(x['chunk_id'] != 5591 and x['chunk_id'] != 5592 for x in rows))

    def test_inputs_and_clinical_boundary(self):
        with self.assertRaises(ValueError):self.reader.audit_sections(0)
        with self.assertRaises(ValueError):self.reader.audit_sections(44,per_section=9)
        rows=self.reader.audit_sections(44,sections=('results',))
        self.assertEqual(rows[0]['status'],'heading_found_not_claim_verified')
        self.assertNotIn('claim_facts_verified',rows[0])
        self.assertTrue(rows[0]['local_open_only'])

if __name__=='__main__':unittest.main()
