import unittest
from evidence_guard import topic_relevant,citation_problem

class GuardTests(unittest.TestCase):
 def test_clozapine_rejects_irrelevant_papers(self):
  q='根据论文，氯氮平在难治性精神分裂症中的疗效如何'
  self.assertFalse(topic_relevant(q,'main.pdf','TAVR and pacemaker implantation after valve surgery'))
  self.assertFalse(topic_relevant(q,'depression.pdf','ketamine in major depressive disorder'))
  self.assertTrue(topic_relevant(q,'schizophrenia.pdf','Clozapine and treatment-resistant schizophrenia'))
 def test_depression_causation(self):
  q='抑郁症的致病因素可能有哪些？'
  self.assertFalse(topic_relevant(q,'mdd.pdf','Antidepressants and treatment dose efficacy for depression.'))
  self.assertTrue(topic_relevant(q,'mdd.pdf','Etiology of major depressive disorder involves complex processes.'))
 def test_invalid_citations(self):
  self.assertEqual(citation_problem('[119] [119] [119]',8),'引用编号超出本次提供的证据范围')
  self.assertIsNotNone(citation_problem('[1] '*25,3))
  self.assertIsNone(citation_problem('结果[1]，局限[2]',2))
if __name__=='__main__':unittest.main()
