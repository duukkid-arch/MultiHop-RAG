import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_medhop import prepare
from score_medhop import score

class TestAdapter(unittest.TestCase):
    def data(self,d):
        one={'id':'a','query':'drug pair?','candidates':['drug A','drug B'],'answer':'drug A','supports':['Paper 1 result.','Paper 2 interaction.']}
        two={'id':'b','query':'other?','candidates':['drug C','drug D'],'answer':'drug D','supports':['Doc.']}
        (d/'train.json').write_text(json.dumps([one]),encoding='utf8')
        (d/'dev.json').write_text(json.dumps([two]),encoding='utf8')
    def test_blind_split_and_offline_score(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.data(p);out=p/'out';prepare(p,out)
            q=(out/'dev_questions_NO_GOLD.jsonl').read_text()
            c=(out/'dev_context_corpus.jsonl').read_text()
            self.assertNotIn('"answer"',q)
            self.assertNotIn('"answer"',c)
            self.assertIn('"corpus_scope"',q)
            self.assertEqual(len((out/'dev_gold_PRIVATE.jsonl').read_text().splitlines()),1)
            pred=p/'pred.jsonl';pred.write_text(json.dumps({'case_id':'medhop_dev_b','prediction':'drug D'})+'\n')
            self.assertEqual(score(out/'dev_gold_PRIVATE.jsonl',pred)['accuracy'],1)
    def test_missing_is_wrong_and_duplicate_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.data(p);out=p/'out';prepare(p,out)
            pred=p/'pred.jsonl';pred.write_text('')
            self.assertEqual(score(out/'dev_gold_PRIVATE.jsonl',pred)['accuracy'],0)
            pred.write_text(('{'+'"case_id":"medhop_dev_b","prediction":"drug D"'+'}\n')*2)
            with self.assertRaises(ValueError):score(out/'dev_gold_PRIVATE.jsonl',pred)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.data(p);out=p/'out';prepare(p,out)
            with self.assertRaises(FileExistsError):prepare(p,out)
    def test_train_dev_overlap_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);self.data(p)
            (p/'dev.json').write_bytes((p/'train.json').read_bytes())
            with self.assertRaises(ValueError):prepare(p,p/'out')

if __name__=='__main__':unittest.main()
